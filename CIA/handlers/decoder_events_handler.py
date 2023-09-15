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
