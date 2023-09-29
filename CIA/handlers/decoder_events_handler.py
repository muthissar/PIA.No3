# import functools
from dataclasses import dataclass
import functools
from typing import Any, Callable, Mapping, Optional, Tuple, Union, Iterable
from warnings import warn
from CIA.handlers.handler import Handler
from CIA.dataloaders.dataloader import DataloaderGenerator
from CIA.utils import (
    all_reduce_scalar,
    is_main_process,
    to_numpy,
    top_k_top_p_filtering,
)
import torch
from tqdm import tqdm
from itertools import islice
import numpy as np
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import einops

# TODO duplicated code with decoder_prefix_handler.py
class DecoderEventsHandler(Handler):
    def __init__(
        self,
        model: DistributedDataParallel,
        model_dir: str,
        dataloader_generator: DataloaderGenerator,
    ) -> None:
        super().__init__(
            model=model, model_dir=model_dir, dataloader_generator=dataloader_generator
        )

    # --- EventsHandler-specific wrappers
    def event_state_to_weight_step(self, output, target_embedded, channel_index):
        return self.model.module.event_state_to_weight_step(
            output, target_embedded, channel_index
        )

    def compute_event_state(self, target, metadata_dict):
        return self.model.module.compute_event_state(
            target, metadata_dict, h_pe_init=None
        )

    # ==== Training methods

    def epoch(
        self,
        data_loader,
        train=True,
        num_batches=None,
    ):
        means = None

        if train:
            self.train()
        else:
            self.eval()

        iterator = enumerate(islice(data_loader, num_batches))
        if is_main_process():
            iterator = tqdm(iterator, ncols=80)

        for sample_id, tensor_dict in iterator:

            # ==========================
            with torch.no_grad():
                x = tensor_dict["x"]
                x, metadata_dict = self.data_processor.preprocess(
                    x, num_events_middle=None
                )

            # ========Train decoder =============
            self.optimizer.zero_grad()
            forward_pass = self.forward(target=x, metadata_dict=metadata_dict)
            loss = forward_pass["loss"]
            # h_pe_init = forward_pass['h_pe'].detach()

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                self.optimizer.step()

            # Monitored quantities
            monitored_quantities = forward_pass["monitored_quantities"]

            # average quantities
            if means is None:
                means = {key: 0 for key in monitored_quantities}
            means = {
                key: value + means[key] for key, value in monitored_quantities.items()
            }

            del loss

        # renormalize monitored quantities
        for key, value in means.items():
            means[key] = all_reduce_scalar(value, average=True) / (sample_id + 1)

        return means

    def inpaint_non_optimized_superconditioning(
        self,
        x,
        metadata_dict,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        num_max_generated_events=None,
        regenerate_first_ts=False,
        null_superconditioning=None,
    ):
        # TODO add arguments to preprocess
        print(f'Placeholder duration: {metadata_dict["placeholder_duration"]}')
        self.eval()
        _, num_events, _ = x.size()

        # TODO only works with batch_size=1 at present
        assert x.size(0) == 1
        index2value = self.dataloader_generator.dataset.index2value
        if null_superconditioning is not None:
            decoding_end = [None] * len(null_superconditioning)
            generated_duration = [0.0] * len(null_superconditioning)
        else:
            decoding_end = [None]
            generated_duration = [0.0]
        placeholder_duration = metadata_dict["placeholder_duration"].item()

        decoding_start_event = metadata_dict["decoding_start"]

        # just to be sure we erase the tokens to be generated
        if not regenerate_first_ts:
            x[:, decoding_start_event:] = 0
        else:
            # Warning, special case when we need to keep the first note!
            x[:, decoding_start_event + 1 :] = 0
            x[:, decoding_start_event, -1] = 0

        # x_null
        if null_superconditioning is not None:
            metadata_dict["placeholder_duration"] = torch.cat(
                [metadata_dict["placeholder_duration"]] * len(null_superconditioning)
            )
            metadata_dict_null = dict(metadata_dict)
            x_null = x.detach().clone()
            placeholder_ind = int(decoding_start_event / 2 - 1)
            sod_ind = decoding_start_event - 1
            # NOTES: -1 is because maybe we still want the last token of prefix to be accessible?
            x_null[:, : placeholder_ind - 1] = self.data_processor.pad_tokens
            x_null[:, placeholder_ind + 1 : sod_ind] = self.data_processor.pad_tokens
            x = torch.cat([x] * len(null_superconditioning))
            x_null = torch.cat([x_null] * len(null_superconditioning))

        if num_max_generated_events is not None:
            num_events = min(
                decoding_start_event + num_max_generated_events, num_events
            )

        done = False

        with torch.no_grad():
            # event_index corresponds to the position of the token BEING generated
            for event_index in range(decoding_start_event, num_events):
                metadata_dict["original_sequence"] = x
                output, target_embedded, _ = self.compute_event_state(
                    target=x,
                    metadata_dict=metadata_dict,
                )
                # extract correct event_step
                output = output[:, event_index]

                if null_superconditioning is not None:
                    metadata_dict_null["original_sequence"] = x_null
                    # output is used to generate auto-regressively all
                    # channels of an event
                    output_null, target_embedded_null, _ = self.compute_event_state(
                        target=x_null,
                        metadata_dict=metadata_dict_null,
                    )
                    output_null = output_null[:, event_index]

                for channel_index in range(self.num_channels_target):
                    # skip updates if we need to only recompute the FIRST TIMESHIFT
                    if (
                        event_index == decoding_start_event
                        and regenerate_first_ts
                        and channel_index < 3
                    ):
                        continue

                    # target_embedded must be recomputed!
                    # TODO could be optimized
                    target_embedded = self.data_processor.embed(x)[:, event_index]
                    weights = self.event_state_to_weight_step(
                        output, target_embedded, channel_index
                    )
                    logits = weights / temperature

                    if null_superconditioning is not None:
                        target_embedded_null = self.data_processor.embed(x_null)[
                            :, event_index
                        ]
                        weights_null = self.event_state_to_weight_step(
                            output_null, target_embedded_null, channel_index
                        )
                        logits_null = weights_null / temperature
                        for batch_index in range(len(logits)):
                            if null_superconditioning[batch_index] != 1:
                                logits[batch_index] = (
                                    logits_null[batch_index]
                                    + (logits[batch_index] - logits_null[batch_index])
                                    * null_superconditioning[batch_index]
                                )

                    # Filter logits
                    filtered_logits = []
                    for logit in logits:
                        filter_logit = top_k_top_p_filtering(
                            logit, top_k=top_k, top_p=top_p
                        )
                        filtered_logits.append(filter_logit)
                    filtered_logits = torch.stack(filtered_logits, dim=0)

                    # Sample from the filtered distribution
                    p = to_numpy(torch.softmax(filtered_logits, dim=-1))

                    # update generated sequence
                    random_state = np.random.get_state()
                    for batch_index in range(len(p)):
                        if decoding_end[batch_index] is not None:
                            continue

                        if event_index >= decoding_start_event:
                            # get new index value
                            np.random.set_state(random_state)

                            new_pitch_index = np.random.choice(
                                np.arange(
                                    self.num_tokens_per_channel_target[channel_index]
                                ),
                                p=p[batch_index],
                            )

                            x[batch_index, event_index, channel_index] = int(
                                new_pitch_index
                            )
                            if (
                                null_superconditioning is not None
                            ):  # write non null value in x_null
                                x_null[batch_index, event_index, channel_index] = int(
                                    new_pitch_index
                                )

                            # check for end symbols
                            end_symbol_index = (
                                self.dataloader_generator.dataset.value2index[
                                    self.dataloader_generator.features[channel_index]
                                ]["END"]
                            )
                            if end_symbol_index == int(new_pitch_index):
                                decoding_end[batch_index] = event_index
                                print(
                                    f"End of decoding due to END symbol generation in batch {batch_index}"
                                )

                            # Additional check:
                            # if the generated duration is > than the
                            # placeholder_duration
                            # TODO hardcoded channel index for timeshifts
                            if channel_index == 3:
                                generated_duration[batch_index] += index2value[
                                    "time_shift"
                                ][new_pitch_index]
                                if (
                                    generated_duration[batch_index]
                                    > placeholder_duration
                                ):
                                    decoding_end[batch_index] = event_index + 1
                                    print(
                                        f"End of decoding due to the generation > than placeholder duration in batch {batch_index}"
                                    )
                                    print(
                                        f"Excess: {generated_duration[batch_index] - placeholder_duration}"
                                    )

                    if all(v is not None for v in decoding_end):
                        done = True
                        break
                if all(v is not None for v in decoding_end):
                    break

            if any(e is None for e in decoding_end):
                done = False
                decoding_end = [num_events if e is None else e for e in decoding_end]
            else:
                done = True

        print(
            f"num events gen: {num_events} - done: {done} - decoding end: {decoding_end}"
        )

        num_event_generated = [e - decoding_start_event for e in decoding_end]
        generated_region = [
            x[batch_index, decoding_start_event : decoding_end[batch_index]]
            for batch_index in range(len(x))
        ]

        return (
            x.cpu(),
            generated_region,
            decoding_end,
            num_event_generated,
            done,
        )

    def inpaint_non_optimized(
        self,
        x,
        metadata_dict,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        num_max_generated_events=None,
        regenerate_first_ts=False,
    ):
        # TODO add arguments to preprocess
        print(f'Placeholder duration: {metadata_dict["placeholder_duration"]}')
        self.eval()
        batch_size, num_events, _ = x.size()

        # TODO only works with batch_size=1 at present
        assert x.size(0) == 1
        index2value = self.dataloader_generator.dataset.index2value
        decoding_end = None
        placeholder_duration = metadata_dict["placeholder_duration"].item()
        generated_duration = 0.0
        decoding_start_event = metadata_dict["decoding_start"]

        # just to be sure we erase the tokens to be generated
        if not regenerate_first_ts:
            x[:, decoding_start_event:] = 0
        else:
            # Warning, special case when we need to keep the first note!
            x[:, decoding_start_event + 1 :] = 0
            x[:, decoding_start_event, -1] = 0

        if num_max_generated_events is not None:
            num_events = min(
                decoding_start_event + num_max_generated_events, num_events
            )

        with torch.no_grad():
            # event_index corresponds to the position of the token BEING generated
            for event_index in range(decoding_start_event, num_events):
                metadata_dict["original_sequence"] = x

                # output is used to generate auto-regressively all
                # channels of an event
                output, target_embedded, h_pe = self.compute_event_state(
                    target=x,
                    metadata_dict=metadata_dict,
                )

                # extract correct event_step
                output = output[:, event_index]

                for channel_index in range(self.num_channels_target):
                    # skip updates if we need to only recompute the FIRST TIMESHIFT
                    if (
                        event_index == decoding_start_event
                        and regenerate_first_ts
                        and channel_index < 3
                    ):
                        continue

                    # target_embedded must be recomputed!
                    # TODO could be optimized
                    target_embedded = self.data_processor.embed(x)[:, event_index]
                    weights = self.event_state_to_weight_step(
                        output, target_embedded, channel_index
                    )
                    logits = weights / temperature

                    # Filter logits
                    filtered_logits = []
                    for logit in logits:

                        filter_logit = top_k_top_p_filtering(
                            logit, top_k=top_k, top_p=top_p
                        )
                        filtered_logits.append(filter_logit)
                    filtered_logits = torch.stack(filtered_logits, dim=0)
                    # Sample from the filtered distribution
                    p = to_numpy(torch.softmax(filtered_logits, dim=-1))

                    # update generated sequence
                    for batch_index in range(batch_size):
                        if event_index >= decoding_start_event:
                            new_pitch_index = np.random.choice(
                                np.arange(
                                    self.num_tokens_per_channel_target[channel_index]
                                ),
                                p=p[batch_index],
                            )
                            x[batch_index, event_index, channel_index] = int(
                                new_pitch_index
                            )

                            end_symbol_index = (
                                self.dataloader_generator.dataset.value2index[
                                    self.dataloader_generator.features[channel_index]
                                ]["END"]
                            )
                            if end_symbol_index == int(new_pitch_index):
                                decoding_end = event_index
                                print("End of decoding due to END symbol generation")

                            # Additional check:
                            # if the generated duration is > than the
                            # placeholder_duration
                            # TODO hardcoded channel index for timeshifts
                            if channel_index == 3:
                                generated_duration += index2value["time_shift"][
                                    new_pitch_index
                                ]
                                if generated_duration > placeholder_duration:
                                    decoding_end = event_index + 1
                                    print(
                                        "End of decoding due to the generation > than placeholder duration"
                                    )
                                    print(
                                        f"Excess: {generated_duration - placeholder_duration}"
                                    )

                    if decoding_end is not None:
                        break
                if decoding_end is not None:
                    break
            if decoding_end is None:
                done = False
                decoding_end = num_events
            else:
                done = True

        num_event_generated = decoding_end - decoding_start_event

        print(
            f"num events gen: {num_events} - done: {done} - decoding end: {decoding_end}"
        )

        generated_region = x[:, decoding_start_event:decoding_end]
        # TODO return everything on GPU
        return x.cpu(), generated_region, decoding_end, num_event_generated, done

    def inpaint(self, x, metadata_dict, temperature=1.0, top_p=1.0, top_k=0):
        # TODO add arguments to preprocess
        print(f'Placeholder duration: {metadata_dict["placeholder_duration"]}')
        self.eval()
        batch_size, num_events, _ = x.size()

        # TODO only works with batch_size=1 at present
        assert x.size(0) == 1

        decoding_end = None
        decoding_start_event = metadata_dict["decoding_start"]
        x[:, decoding_start_event:] = 0
        with torch.no_grad():
            # event_index corresponds to the position of the token BEING generated
            for event_index in range(decoding_start_event, num_events):
                metadata_dict["original_sequence"] = x

                # output is used to generate auto-regressively all
                # channels of an event
                output, target_embedded, h_pe = self.compute_event_state(
                    target=x,
                    metadata_dict=metadata_dict,
                )

                # extract correct event_step
                output = output[:, event_index]

                for channel_index in range(self.num_channels_target):
                    # target_embedded must be recomputed!
                    # TODO could be optimized
                    target_embedded = self.data_processor.embed(x)[:, event_index]
                    weights = self.event_state_to_weight_step(
                        output, target_embedded, channel_index
                    )
                    logits = weights / temperature

                    filtered_logits = []
                    for logit in logits:
                        filter_logit = top_k_top_p_filtering(
                            logit, top_k=top_k, top_p=top_p
                        )
                        filtered_logits.append(filter_logit)
                    filtered_logits = torch.stack(filtered_logits, dim=0)
                    # Sample from the filtered distribution
                    p = to_numpy(torch.softmax(filtered_logits, dim=-1))

                    # update generated sequence
                    for batch_index in range(batch_size):
                        if event_index >= decoding_start_event:
                            new_pitch_index = np.random.choice(
                                np.arange(
                                    self.num_tokens_per_channel_target[channel_index]
                                ),
                                p=p[batch_index],
                            )
                            x[batch_index, event_index, channel_index] = int(
                                new_pitch_index
                            )

                            end_symbol_index = (
                                self.dataloader_generator.dataset.value2index[
                                    self.dataloader_generator.features[channel_index]
                                ]["END"]
                            )
                            if end_symbol_index == int(new_pitch_index):
                                decoding_end = event_index

                    if decoding_end is not None:
                        break
                if decoding_end is not None:
                    break
            if decoding_end is None:
                done = False
                decoding_end = num_events
            else:
                done = True

        num_event_generated = decoding_end - decoding_start_event
        return x.cpu(), decoding_end, num_event_generated, done
    
    def inpaint_ic_curve(
        self,
        x,
        metadata_dict,
        interpolation_time_points,
        k_traces,
        weight_fn,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        num_max_generated_events=None,
        regenerate_first_ts=False,
    ):
        # TODO add arguments to preprocess
        print(f'Placeholder duration: {metadata_dict["placeholder_duration"]}')
        self.eval()
        batch_size, num_events, _ = x.size()
        # TODO only works with batch_size=1 at present
        assert x.size(0) == 1
        index2value = self.dataloader_generator.dataset.index2value
        decoding_end = None
        placeholder_duration = metadata_dict["placeholder_duration"].item()
        generated_duration = torch.zeros(k_traces, x.shape[1])
        decoding_start_event = metadata_dict["decoding_start"]
        # NOTE: copy the original sequence:
        x = x.detach().clone()
        metadata_dict['original_sequence'] = x.detach().clone()
        timepoints_template, ic_template = self.compute_token_onsets(
            x=x,
            metadata_dict=metadata_dict
        )
        warn('Most likely, we would actually need to start sampling with a shift, if first note should not always align.')
        interpolator_template = DecoderEventsHandler.gen_interpolate(
            ic_times=[timepoints_template],
            ics=[ic_template],
            weight_fn=weight_fn
        )
        # just to be sure we erase the tokens to be generated
        # if not regenerate_first_ts:
        #     x[:, decoding_start_event:] = 0
        # else:
        #     # Warning, special case when we need to keep the first note!
        #     x[:, decoding_start_event + 1 :] = 0
        #     x[:, decoding_start_event, -1] = 0
        warn('It does not make sense to limit the number of tokens that we can resample.')
        if num_max_generated_events is not None:
            num_events = min(
                decoding_start_event + num_max_generated_events, num_events
            )
        # NOTE: preallocate data structures
        batch_indices = torch.tensor([0])
        event_indices = torch.tensor(k_traces*[decoding_start_event])
        x = einops.repeat(x, '1 ... -> k ...', k=k_traces).contiguous()
        ics = torch.zeros_like(x, dtype=torch.float32,device='cpu')
        done = torch.zeros(k_traces, dtype=torch.bool, device='cpu')
        # warn('Next two lines only for debugging purposes...')
        # batch_indices = torch.tensor([0,0])
        # event_indices = torch.tensor([decoding_start_event, decoding_start_event])
        timepoint_idx = 0
        first_time_expand = True
        first_index = 0
        with torch.no_grad():
            # event_index corresponds to the position of the token BEING generated
            # for event_index in range(decoding_start_event, num_events):
            warn('Change while true condition.')
            while True:
                if first_time_expand:
                    # NOTE indices the sequences which did not yet exceed the timepoint prune limit
                    batch_indices = torch.arange(k_traces)
                    # TODO: remove contiguous
                    event_indices[:] = einops.repeat(event_indices[first_index], '-> k', k=k_traces)
                    generated_duration[:] = einops.repeat(generated_duration[first_index], '... -> k ...', k=k_traces)
                    # unexceeded_timepoint = []
                    x[:] = einops.repeat(x[first_index], '... -> k ...', k=k_traces)
                    ics[:] = einops.repeat(ics[first_index], '... -> k ...', k=k_traces)
                    first_time_expand = False
                while len(batch_indices) > 0:
                    max_idx = event_indices[batch_indices].max()
                    x_ = x[batch_indices, :max_idx].contiguous()
                    metadata_dict["original_sequence"] = x_.clone()
                    # output is used to generate auto-regressively all
                    # channels of an event
                    warn('Check that metadata_dict is correctly used...')
                    output_, target_embedded, h_pe = self.compute_event_state(
                        target=x_,
                        metadata_dict=metadata_dict,
                    )

                    # extract correct event_step
                    output = output_[torch.arange(len(batch_indices)), event_indices[batch_indices]]

                    unexceeded_timepoint = []
                    for channel_index in range(self.num_channels_target):    
                        # skip updates if we need to only recompute the FIRST TIMESHIFT
                        if (
                            (event_indices[batch_indices] == decoding_start_event).any()
                            and regenerate_first_ts
                            and channel_index < 3
                        ):
                            raise NotImplementedError('Check the functionality is meaningful for ic curves')
                            continue

                        # target_embedded must be recomputed!
                        # TODO could be optimized
                        target_embedded = self.data_processor.embed(x[batch_indices])[
                            torch.arange(len(batch_indices)),
                            event_indices[batch_indices]
                        ]
                        weights = self.event_state_to_weight_step(
                            output, target_embedded, channel_index
                        )
                        logits = weights / temperature

                        # Filter logits
                        # filtered_logits = []
                        # for logit in logits:

                        #     filter_logit = top_k_top_p_filtering(
                        #         logit, top_k=top_k, top_p=top_p
                        #     )
                        #     filtered_logits.append(filter_logit)
                        # filtered_logits = torch.stack(filtered_logits, dim=0)
                        filtered_logits = logits
                        # Sample from the filtered distribution
                        p = to_numpy(torch.softmax(filtered_logits, dim=-1))
                        # update generated sequence
                        for p_, batch_index, event_index in zip(p, batch_indices,  event_indices[batch_indices]):
                            # TODO: this check seems to be allways true, since we start with event_index == decoding_start_event
                            if event_index >= decoding_start_event:
                                new_pitch_index = np.random.choice(
                                    np.arange(
                                        self.num_tokens_per_channel_target[channel_index]
                                    ),
                                    p=p_,
                                )
                                x[batch_index, event_index, channel_index] = int(
                                    new_pitch_index
                                )
                                ics[batch_index, event_index, channel_index] = p_[new_pitch_index].tolist()

                                end_symbol_index = (
                                    self.dataloader_generator.dataset.value2index[
                                        self.dataloader_generator.features[channel_index]
                                    ]["END"]
                                )
                                if end_symbol_index == int(new_pitch_index):
                                    warn('Find out if end can happen accross different channels?')
                                    # NOTE: allow to stop the sequence if its "80% done"
                                    done_pct = 0.8
                                    if interpolation_time_points[timepoint_idx]/interpolation_time_points[-1] >=done_pct:    
                                        done[batch_index] = True
                                        decoding_end = event_index
                                        print("End of decoding due to END symbol generation")

                                # Additional check:
                                # if the generated duration is > than the
                                # placeholder_duration
                                # TODO hardcoded channel index for timeshifts
                                if channel_index == 3:
                                    generated_duration[batch_index, event_index] = generated_duration[batch_index, event_index - 1] + index2value["time_shift"][
                                        new_pitch_index
                                    ]
                                    if generated_duration[batch_index, event_index] < interpolation_time_points[timepoint_idx]:
                                        unexceeded_timepoint.append(batch_index)
                                    if generated_duration[batch_index, event_index] > placeholder_duration:
                                        raise NotImplementedError
                                        decoding_end = event_index + 1
                                        print(
                                            "End of decoding due to the generation > than placeholder duration"
                                        )
                                        print(
                                            f"Excess: {generated_duration - placeholder_duration}"
                                        )
                        if decoding_end is not None:
                            break
                    event_indices[batch_indices] += 1    
                    batch_indices = torch.LongTensor(unexceeded_timepoint)
                    if decoding_end is not None:
                        break 
                ic_times_list = [dur[decoding_start_event-1:event_index-1] for dur, event_index in zip(generated_duration, event_indices)]
                ics_list = [ic[decoding_start_event:event_index] for ic,event_index in zip(ics, event_indices)]
                interpolator = DecoderEventsHandler.gen_interpolate(
                    ic_times = ic_times_list,
                    ics = ics_list,
                    weight_fn = DecoderEventsHandler.mov_avg,
                )
                ts = interpolation_time_points[timepoint_idx:timepoint_idx+1]
                int_time = interpolator(ts)
                int_time_temp = interpolator_template(ts)
                # NOTE: for now just 
                abs_diffs = (int_time.sum(2) - int_time_temp.sum(2)).abs().sum(dim=(1))
                _, min_id = abs_diffs.min(dim=0)

                # NOTE: store pruned in first 
                batch_indices = torch.tensor([first_index])
                event_indices[first_index] = event_indices[min_id]
                generated_duration[first_index] = generated_duration[min_id]
                # unexceeded_timepoint = []
                x[first_index] = x[min_id]
                ics[first_index] = ics[min_id]
                first_time_expand = True
                timepoint_idx += 1
                if timepoint_idx == len(interpolation_time_points):
                    decoding_end = event_indices[first_index].item()


                    break
            warn('What to do here?')
            # if decoding_end is None:
            #     done = False
            #     decoding_end = num_events
            # else:
            #     done = True
        warn('What to do here?')
        # num_event_generated = decoding_end - decoding_start_event

        # print(
        #     f"num events gen: {num_events} - done: {done} - decoding end: {decoding_end}"
        # )

        # generated_region = x[first_index, decoding_start_event:decoding_end][None].cpu()
        # # TODO return everything on GPU
        # return x[first_index].cpu(), generated_region, decoding_end, num_event_generated, done
        @dataclass
        class ICRes:
            tok: torch.Tensor
            ic_tok: torch.Tensor
            ic_int: torch.Tensor
            timepoints: torch.Tensor
            decoding_end: Optional[int] = None
        temp = ICRes(
            metadata_dict['original_sequence'],
            ic_template,
            interpolator_template(interpolation_time_points)[0],
            timepoints_template
        )
        gen = ICRes(
            x[first_index].cpu(),
            ics[first_index].cpu(),
            interpolator(interpolation_time_points)[first_index],
            ic_times_list[first_index],
            decoding_end
        )
        return temp, gen
        # return x[first_index].cpu(), ics[first_index].cpu(), ic_template,\
        #     interpolator(interpolation_time_points)[first_index], \
        #         interpolator_template(interpolation_time_points)[0], decoding_end
    @staticmethod
    def integration(a : float, b : float, unique_timepoint : torch.Tensor, cum_ics: torch.Tensor): 
            assert cum_ics.dim() == 2
            res = torch.zeros_like(cum_ics[0])
            for t, ic in zip(unique_timepoint, cum_ics):
                if t >= a and t <= b:
                    res += ic
                if t >= b:
                    break
            return res
    
    @staticmethod
    def mov_avg(time_diffs : torch.FloatTensor) -> torch.Tensor:
        # NOTE: (bz, T, tokens)
        window_size = 0.5
        c = 0.
        mov_avg = (-c*time_diffs).exp()
        mov_avg[time_diffs>=window_size] = 0
        return mov_avg
    @staticmethod
    def gen_interpolate(
            ic_times : Iterable[torch.FloatTensor],
            ics: Iterable[torch.FloatTensor],
            weight_fn: Callable[[torch.FloatTensor], torch.FloatTensor]
        ) ->  Callable[[torch.FloatTensor], torch.FloatTensor]:
        lens = [len(ic) for ic in ics]
        assert all(l == len(ic) for l, ic in zip(lens, ics))
        ic_times = torch.nn.utils.rnn.pad_sequence(ic_times, batch_first=True)
        ics = torch.nn.utils.rnn.pad_sequence(ics, batch_first=True)
        def interpolate(t : torch.FloatTensor) -> torch.FloatTensor:
            # NOTE: a matrix of (bz, T, TOKENS)
            time_diffs = t[None, :, None] - ic_times[:, None]
            w = weight_fn(time_diffs)
            w[time_diffs <.0] = 0.
            # NOTE: automatically the padding cancels automatically because of the ic padding.
            return einops.einsum(w, ics, 'bz T tok, bz tok ... -> bz T ...')
        return interpolate
    def quant(
            xs: Union[float, int, Iterable],
            interpolator : Callable[[float, float], float],
            end = Optional[float],
        ):
        if isinstance(xs, Iterable):
            ys = torch.tensor(xs)
        else:
            assert end
            if isinstance(xs, float):
                step_size = xs
                ys = torch.arange(0, end+(step_size-1e-9), step_size)
            elif isinstance(xs, int):
                steps = xs
                ys = torch.linspace(0, end, steps)
            else:
                raise ValueError
        return ys, torch.stack([interpolator(ys[i], ys[i+1]) for  i in range(len(ys)-1)],axis=0)

    def compute_ic_template(
        self,
        x,
        metadata_dict,
        # temperature=1.0,
        # top_p=1.0,
        # top_k=0,
        # num_max_generated_events=None,
        # regenerate_first_ts=False,
    ):
        

        unique_timepoint, cum_ics = self.compute_token_onsets(x, metadata_dict)
        integrator = functools.partial(self.integration, unique_timepoint=unique_timepoint, cum_ics=cum_ics)
        return integrator
        
    def compute_token_onsets(
        self,
        x : torch.Tensor,
        metadata_dict : dict,
        match_original_onsets : Optional[Mapping[int, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO add arguments to preprocess
        self.eval()
        batch_size, num_events, _ = x.size()

        # TODO only works with batch_size=1 at present
        assert batch_size == 1
        with torch.no_grad():
            # self.forward(target=x, metadata_dict=metadata_dict)
            output, target_embedded, h_pe = self.model.module.compute_event_state(
                x, metadata_dict, h_pe_init=None
            )

            # auto regressive predictions from output
            weights_per_category = self.model.module.event_state_to_weights(
                output=output, target_embedded=target_embedded
            )
            ics = []
            # NOTE: only works for batch_size == 1
            for w, x_, mask in zip(weights_per_category, x.permute(2,0,1), (~metadata_dict['loss_mask']).permute(2,0,1)):
                w = w[mask]
                x_ = x_[mask]
                ic = torch.nn.functional.cross_entropy(input=w, target=x_, reduction='none')
                ic = einops.rearrange(ic, '(b n) -> b n', b=batch_size)
                ics.append(ic.cpu())
            ics = torch.stack(ics, dim=-1)
            middle_tokens = x[~metadata_dict['loss_mask']].view(batch_size, -1 , self.num_channels_target)
            if match_original_onsets is not None:
                onsets = torch.tensor(np.append(match_original_onsets, match_original_onsets[:, -1:], axis=1))
            else:
                # TODO: There's some small deviation between this calculation and the one obtained by simply using the cum
                onsets = torch.cumsum(torch.tensor(
                    [0]+[self.dataloader_generator.dataset.index2value['time_shift'][tok[3].item()] for tok in middle_tokens[0,:-1]]
                ), dim=0
                )[None]
                # cum_shifts = self.dataloader_generator.get_elapsed_time(middle_tokens)
                # onsets = cum_shifts.roll(1)
                # onsets[:, 0] = 0
            # TODO: check if this is what we want...
            unique_timepoint, cum_ics = DecoderEventsHandler.unique_timepoints(onsets[0], ics[0])
            
            # TODO: remove batches everywhere because it's not working anyway
            return unique_timepoint, cum_ics
    @staticmethod
    def unique_timepoints(onsets : torch.Tensor, ics : torch.Tensor):
        assert ics.dim() == 2 and onsets.dim() == 1, "Batched not implimented..."
        timepoints = []
        for onset, ic in zip(onsets, ics):
            onset = onset.item()
            if timepoints and timepoints[-1][0] == onset:
                timepoints[-1][1] = timepoints[-1][1] + ic
            else:
                timepoints.append([onset, ic])
        unique_timepoint, cum_ics = zip(*timepoints)
        unique_timepoint = torch.tensor(unique_timepoint)
        cum_ics = torch.stack(cum_ics, dim=0)
        return unique_timepoint, cum_ics