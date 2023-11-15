import functools
import os
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union, Iterable
from warnings import warn
from CIA.handlers.handler import Handler
from CIA.dataloaders.dataloader import DataloaderGenerator
# from CIA.ic import gen_interpolate
from CIA.ic import ICCurve, ICRes, Interpolator, Piece, unique_timepoints, numerial_stable_softmax_entr
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
# import logging
import multiprocessing
logger = multiprocessing.get_logger()

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
        piece: Piece,
        interpolator_template : Optional[torch.FloatTensor] = None,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        num_max_generated_events=None,
        regenerate_first_ts=False,
    ):
        # TODO add arguments to preprocess
        # print(f'Placeholder duration: {metadata_dict["placeholder_duration"]}')
        logger.debug(f'Placeholder duration: {metadata_dict["placeholder_duration"]}')
        self.eval()
        batch_size, num_events, n_feats = x.size()
        # TODO only works with batch_size=1 at present
        assert x.size(0) == 1
        index2value = self.dataloader_generator.dataset.index2value
        # decoding_end = None
        placeholder_duration = metadata_dict["placeholder_duration"].item()
        generated_duration = torch.zeros(k_traces, x.shape[1])
        decoding_start_event = metadata_dict["decoding_start"]
        template_decoding_end = metadata_dict["decoding_end"].item()
        # NOTE: copy the original sequence:
        original_sequence = x.detach().cpu().clone()
        x = x.detach().clone()
        # TODO: deprecate all instances of setting original_sequence
        # metadata_dict['original_sequence'] = x.detach().clone()
        # NOTE: this is only for the middle tokens!
        if interpolator_template is None:
            timepoints_tok_template, ic_tok_template, entr_template = self.compute_token_onsets(
                x=metadata_dict['original_sequence'],
                metadata_dict=metadata_dict
                )
            # warn('Most likely, we would actually need to start sampling with a shift, if first note should not always align.')
            logger.warning('Most likely, we would actually need to start sampling with a shift, if first note should not always align.')
            interpolator_template = Interpolator(
                ic_times=[timepoints_tok_template],
                ics=[ic_tok_template],
                weight_fn=weight_fn
            )
        else:
            timepoints_tok_template = None
            ic_tok_template = None
        # just to be sure we erase the tokens to be generated
        # if not regenerate_first_ts:
        #     x[:, decoding_start_event:] = 0
        # else:
        #     # Warning, special case when we need to keep the first note!
        #     x[:, decoding_start_event + 1 :] = 0
        #     x[:, decoding_start_event, -1] = 0
        # warn('It does not make sense to limit the number of tokens that we can resample.')
        logger.warning('It does not make sense to limit the number of tokens that we can resample.')
        if num_max_generated_events is not None:
            num_events = min(
                decoding_start_event + num_max_generated_events, num_events
            )
        # NOTE: preallocate data structures
        batch_indices = torch.tensor([0])
        event_indices = torch.tensor(k_traces*[decoding_start_event])
        x = einops.repeat(x, '1 ... -> k ...', k=k_traces).contiguous()
        ics = torch.zeros_like(x, dtype=torch.float32,device='cpu')
        entrs = torch.zeros_like(x, dtype=torch.float32,device='cpu')
        done = torch.zeros((k_traces, n_feats), dtype=torch.bool, device='cpu')
        # warn('Next two lines only for debugging purposes...')
        # batch_indices = torch.tensor([0,0])
        # event_indices = torch.tensor([decoding_start_event, decoding_start_event])
        timepoint_idx = 0
        first_time_expand = True
        best_index = 0
        with torch.no_grad():
            # event_index corresponds to the position of the token BEING generated
            # for event_index in range(decoding_start_event, num_events):
            rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
            with tqdm(total=len(interpolation_time_points), position=rank+2, desc=f'Sampling rank: {rank}') as pbar:
                while True:
                    if first_time_expand:
                        # NOTE indices the sequences which did not yet exceed the timepoint prune limit
                        # start_time = time.time()
                        not_done = ~done.any(-1)
                        batch_indices = torch.arange(k_traces)[not_done]
                        event_indices[not_done] = event_indices[best_index]
                        generated_duration[not_done] = generated_duration[best_index]
                        x[not_done] = x[best_index]
                        ics[not_done] = ics[best_index]
                        entrs[not_done] = entrs[best_index]
                        first_time_expand = False
                        # print(f"expand time: {time.time()-start_time}")
                    while len(batch_indices) > 0:
                        # TODO: why does it only work with full sequence lenght?
                        # max_idx = event_indices[batch_indices].max()
                        max_idx = x.size(1)
                        x_ = x[batch_indices, :max_idx].contiguous()
                        metadata_dict["original_sequence"] = x_.clone()
                        # output is used to generate auto-regressively all
                        # channels of an event
                        # start_time = time.time()
                        output_, target_embedded, h_pe = self.compute_event_state(
                            target=x_,
                            metadata_dict=metadata_dict,
                        )
                        # print(f"state time: {time.time()-start_time}")
                        # NOTE: for longer generation, we need to do new preprocessing.
                        # TODO: define hop length and then reprocess. However, how will this affect the placeholder duration?
                        # Is it possible to simply minus the already generated time, and how does it work, when the placeholder duration is longer than the ones in
                        # the training data? Maybe we can simply use the highest value that was use in the train set, but how does this work in reality????
                        output = output_[torch.arange(len(batch_indices)), event_indices[batch_indices]]

                        unexceeded_timepoint = []
                        # start_time = time.time()
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
                            for p_, batch_index, event_index, logits_ in zip(p, batch_indices,  event_indices[batch_indices], filtered_logits):
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
                                    # ics[batch_index, event_index, channel_index] = p_[new_pitch_index].tolist()
                                    ics[batch_index, event_index, channel_index] = (-np.log(p_[new_pitch_index])).tolist()
                                    entrs[batch_index, event_index, channel_index] = numerial_stable_softmax_entr(logits_, dim=-1).sum(dim=-1)

                                    end_symbol_index = (
                                        self.dataloader_generator.dataset.value2index[
                                            self.dataloader_generator.features[channel_index]
                                        ]["END"]
                                    )
                                    # TODO: move all termination checks together, for better readability.
                                    if end_symbol_index == int(new_pitch_index):
                                        # warn('Find out if end can happen accross different channels?')
                                        # logger.warn('Find out if end can happen accross different channels?')
                                        # NOTE if a sequence is done, we keep it's interpolation and continue
                                        # computing the rest of the timepoints
                                        # done_pct = 0.8
                                        # NOTE: this is 
                                        # if interpolation_time_points[timepoint_idx]/interpolation_time_points[-1] >=done_pct:
                                        done[batch_index, channel_index] = True
                                        # NOTE: avoid end token to be written in the middle tokens
                                        event_indices[batch_indices] -= 1
                                        # decoding_end = event_index
                                        logger.info("End of decoding due to END symbol generation")

                                    # Additional check:
                                    # if the generated duration is > than the
                                    # placeholder_duration
                                    # TODO hardcoded channel index for timeshifts
                                    if channel_index == 3:
                                        shift = index2value["time_shift"][
                                            new_pitch_index
                                        ]
                                        shift = 0.0 if shift == 'END' else shift
                                        generated_duration[batch_index, event_index] = generated_duration[batch_index, event_index - 1] + shift
                                        if event_index == x.size(1) - 1:
                                            # print("End of decoding due to reaching last sequence index")
                                            # print(
                                            #     f"Missing: {generated_duration[batch_index, event_index] - placeholder_duration}"
                                            # )
                                            logger.debug(f"End of decoding due to reaching last sequence index.\nMissing: {generated_duration[batch_index, event_index] - placeholder_duration}")
                                            done[batch_index, channel_index] = True
                                        elif generated_duration[batch_index, event_index] > placeholder_duration:
                                            # raise NotImplementedError
                                            # decoding_end = event_index + 1
                                            # print(
                                            #     "End of decoding due to the generation > than placeholder duration"
                                            # )
                                            # print(
                                            #     f"Excess: {generated_duration[batch_index, event_index] - placeholder_duration}"
                                            # )
                                            logger.debug('End of decoding due to the generation > than placeholder duration.\nExcess: {generated_duration[batch_index, event_index] - placeholder_duration}')
                                            done[batch_index, channel_index] = True
                                        elif generated_duration[batch_index, event_index] < interpolation_time_points[timepoint_idx]:
                                            unexceeded_timepoint.append(batch_index)
                        # print(f"events time: {time.time()-start_time}")
                        event_indices[batch_indices] += 1    
                        batch_indices = torch.LongTensor(unexceeded_timepoint)
                        # if x.size(1) in event_indices:
                        #     raise NotImplementedError("this messes up if the sequence is exceeded in event_indices.")
                        # if decoding_end is not None:
                        #     break
                    # TODO: we can optimize this by only computing for the ones actively expanded (i.e. in batch_index),
                    # start_time = time.time()
                    ic_times_list = [dur[decoding_start_event-1:event_index-1] for dur, event_index in zip(generated_duration, event_indices)]
                    ics_list = [ic[decoding_start_event:event_index] for ic,event_index in zip(ics, event_indices)]
                    interpolator = Interpolator(
                        ic_times = ic_times_list,
                        ics = ics_list,
                        weight_fn = weight_fn,
                    )
                    ts = interpolation_time_points[timepoint_idx:timepoint_idx+1]
                    int_time = interpolator(ts)
                    int_time_temp = interpolator_template(ts)
                    # NOTE: for now we just compute the abs of the sum of all channels
                    abs_diffs = (int_time.sum(2) - int_time_temp.sum(2)).abs().sum(dim=(1))
                    _, best_index_all = abs_diffs.min(dim=0)
                    # TODO: remove the termination from here to reduce spaghetti code
                    if done.any(-1).all() or timepoint_idx == len(interpolation_time_points) - 1:
                        decoding_end = event_indices[best_index_all].item()
                        # NOTE: to allign with the decoding_end which is pointing to the end token... 
                        decoding_end +=1
                        break
                    abs_diffs[done.any(-1)] = float('inf')
                    _, best_index = abs_diffs.min(dim=0)
                    if done[best_index_all].any():
                        done[:] = False    
                        done[best_index_all] = True
                    else:
                        # in this case as the finished sequence is worse than another sequence, don't keep any.
                        done[:] = False    

                    # NOTE: store pruned in first 
                    # batch_indices = torch.tensor([best_index])
                    # event_indices[first_index] = event_indices[min_id]
                    # generated_duration[first_index] = generated_duration[min_id]
                    # unexceeded_timepoint = []
                    # x[first_index] = x[min_id]
                    # ics[first_index] = ics[min_id]
                    first_time_expand = True
                    # warn("while do until it's above the threshold")
                    # TODO: we could do something like rejection sampling, or use some heuristic search.
                    # logger.warning("while do until it's above the threshold")
                    timepoint_idx += 1
                    # NOTE: for some reason as soon as we get to here, it starts to be slow.
                    # its likely to be caused by cache misses.
                    # answ : no speedup is due to reducing batch index, such that fewer and fewer samples are computed...
                    # print(f"interpolate: {time.time()-start_time}")
                    pbar.update()
                    # if timepoint_idx == len(interpolation_time_points):
        # num_event_generated = decoding_end - decoding_start_event

        # print(
        #     f"num events gen: {num_events} - done: {done} - decoding end: {decoding_end}"
        # )

        # generated_region = x[first_index, decoding_start_event:decoding_end][None].cpu()
        # # TODO return everything on GPU
        # return x[first_index].cpu(), generated_region, decoding_end, num_event_generated, done
        temp = ICRes(
            tok = original_sequence[0],
            ic_tok = ic_tok_template,
            entr_tok = entr_template,
            timepoints = timepoints_tok_template,
            ic_int = interpolator_template(interpolation_time_points)[0],
            timepoints_int = interpolation_time_points,
            decoding_end=template_decoding_end,
            piece = piece
        )
        ic_tok_gen = ics[best_index,decoding_start_event:decoding_end-1].cpu()
        entr_gen = entrs[best_index,decoding_start_event:decoding_end-1].cpu()
        gen = ICRes(
            tok = x[best_index_all].cpu(),
            # ics[best_index].cpu(),
            ic_tok = ic_tok_gen,
            entr_tok = entr_gen,
            timepoints = ic_times_list[best_index_all],
            ic_int = interpolator(interpolation_time_points)[best_index_all],
            timepoints_int = interpolation_time_points,
            decoding_end=decoding_end,
            piece = piece
        )
        return temp, gen
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
        

        unique_timepoint, cum_ics, entr = self.compute_token_onsets(x, metadata_dict)
        integrator = functools.partial(self.integration, unique_timepoint=unique_timepoint, cum_ics=cum_ics)
        return integrator
        
    def compute_token_onsets(
        self,
        x : torch.Tensor,
        metadata_dict : dict,
        # match_original_onsets : Optional[Mapping[int, Any]] = None
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
            # NOTE: only works for batch_size == 1
            ics = torch.empty_like(x, device='cpu', dtype=torch.float32)
            entrs = torch.empty_like(x, device='cpu', dtype=torch.float32)
            for i, (w, x_, mask) in enumerate(zip(weights_per_category, x.permute(2,0,1), (~metadata_dict['loss_mask']).permute(2,0,1))):
                w = w[mask]
                x_ = x_[mask]
                ic = torch.nn.functional.cross_entropy(input=w, target=x_, reduction='none')
                ic = einops.rearrange(ic, '(b n) -> b n', b=batch_size)
                ics[..., i][mask] = ic.cpu()
                entr = numerial_stable_softmax_entr(w,dim=-1).sum(dim=-1)
                entrs[..., i][mask] = entr.cpu()
                
            # NOTE: again the decoding_end could be problematic if it includes patting?
            middle_slice = slice(metadata_dict['decoding_start'], metadata_dict['decoding_end']-1)
            middle_tokens = x[0, middle_slice]
            # TODO: There's some small deviation between this calculation and the one obtained by simply using the cum
            onsets = torch.cumsum(torch.tensor(
                [0]+[self.dataloader_generator.dataset.index2value['time_shift'][tok[3].item()] for tok in middle_tokens]
                , dtype=torch.float32), dim=0
            )
            # TODO: check if this is what we want...
            ics_middle = ics[0, middle_slice]
            entrs_middle = entrs[0, middle_slice]
            # NOTE: it seems uneccesary to compute unique timepoints, rather we can simply 


            # unique_timepoint, cum_ics = unique_timepoints(onsets, ics_middle)
            onsets = onsets[:-1]
            # TODO: remove batches everywhere because it's not working anyway
            return onsets, ics_middle, entrs_middle
