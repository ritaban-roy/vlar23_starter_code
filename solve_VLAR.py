#!/usr/bin/env python3
"""
    VLAR Challenge: Submission Code Demo.
    ************************************
    
    This is a starter code demonstrating how to format your pre-trained model for evaluation. 
    It shows where to expect to read the test puzzle images from, and how to produce 
    the output, which can be evaluated using our evaluation code. 
    
    Please see the predict_on_challenge_data() function below for details. Formally, the code shows four steps:
        1) To read the puzzles (see data_loader.py, SMART_Challenge_Data class)
        2) Get/load the prediction/solution model: get_SMART_solver_model())
        3) Run the prediction model on the test puzzles and collect responses: make_predictions()
        4) Collect the responses in a json file for evaluation: make_response_json()
    
    For this demo, we provide a pretrained ResNet-50 + BERT pre-trained model traiend
    on the SMART-101 dataset in the puzzle_split mode. This model is provided in ./checkpoints/ckpt_resnet50_bert_212.pth
    
    See scripts.sh file for the commandlines to train the model on SMART-101 dataset and how to run the model on the VLAR challenge
    val and test datasets. 
    
    Specifically, note that the VLAR-val.json and VLAR-test.json files containing the VLAR challenge puzzles
    are assumed to be kept in /dataset/ folder, and a method should write the responses to /submission/submission.json
    as described in make_predictions() below. 
    
    Note
    ----
    In this demo, we do not use the answer candidate options within the model. However, 
    a user may chose to have additional inputs to the model for taking in the options.
    
    For questions: contact the VLAR organizers at vlariccv23@googlegroups.com
"""
import torch
import numpy as np
import json
import os
import time
import net
import data_loader as dl
import globvars as gv

import transformers
from peft import PeftModelForCausalLM, PeftConfig, AutoPeftModelForCausalLM
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig

class CustomLLaVAModel(LlavaForConditionalGeneration):
  def __init__(self, config):
    super().__init__(config)
    self.word_embeddings = self.get_input_embeddings()

class PromptBeforeInstruction(PeftModelForCausalLM):
  def __init__(self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default") -> None:
      super().__init__(model, peft_config, adapter_name)
  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      inputs_embeds=None,
      labels=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
      token_type_ids=None,
      task_ids=None,
      pixel_values=None,
      position_ids=None,
      past_key_values=None,
      vision_feature_layer=-2,
      vision_feature_select_strategy="default",
      use_cache=None,
      **kwargs,
  ):
    peft_config = self.active_peft_config
    batch_size = input_ids.shape[0]
    if attention_mask is not None:
      prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
      kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)
    kwargs.update(
        {
            "attention_mask": attention_mask,
            "labels": labels,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
    )
    # 1. Extra the input embeddings
    insert_idx = torch.where(input_ids[0] == self.base_model.config.image_token_index)[0][0].item()
    inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

    # 2. Get prompt embedding and labels
    if labels is not None:
      prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
      kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
    prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
    prompts = prompts.to(inputs_embeds.dtype)
    inputs_embeds = torch.cat((inputs_embeds[:, :insert_idx+1], prompts[:, :peft_config.num_virtual_tokens], inputs_embeds[:, insert_idx+1:]), dim=1)
    # 3. Merge text and images and trainable prompt
    if pixel_values is not None and input_ids.shape[1] != 1:
        image_outputs = self.base_model.vision_tower(pixel_values, output_hidden_states=True)
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                f"Unexpected select feature strategy: {self.base_model.config.vision_feature_select_strategy}"
            )

        image_features = self.base_model.multi_modal_projector(selected_image_feature)
        inputs_embeds, attention_mask, labels, position_ids = self.base_model._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, labels
        )
        if labels is None:
            labels = torch.full_like(attention_mask, self.base_model.config.ignore_index).to(torch.long)
    outputs = self.base_model.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    logits = outputs[0]

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:]
            shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    out = dict()
    out["loss"] = loss
    out["logits"] = logits
    out["past_key_values"] = outputs.past_key_values
    out["hidden_states"] = outputs.hidden_states
    out["attentions"] = outputs.attentions
    return out

  def prepare_inputs_for_generation(
        self, input_ids=None, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None, task_ids=None, **kwargs
    ):
    peft_config = self.active_peft_config
    model_kwargs = self.base_model_prepare_inputs_for_generation(input_ids, **kwargs)

    # https://github.com/huggingface/transformers/pull/26681/ introduced new cache format
    # for some architectures which requires a special fix for prompt tuning etc.
    # TODO: starting with transformers 4.38, all architectures should support caching.
    uses_transformers_4_38 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.38.0")
    uses_transformers_4_36 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.36.0")
    transformers_new_cache_archs = ["llama", "mistral", "persimmon", "phi"]
    uses_cache = uses_transformers_4_38 or (
        uses_transformers_4_36 and self.base_model.config.model_type in transformers_new_cache_archs
    )

    if uses_cache and (model_kwargs["past_key_values"] is not None):
        # change in the logic of `prepare_inputs_for_generation` makes the below code necessary
        # In prompt learning methods, past key values are longer when compared to the `input_ids`.
        # As such only consider the last input ids in the autogressive generation phase.
        if model_kwargs["past_key_values"][0][0].shape[-2] >= model_kwargs["input_ids"].shape[1]:
            model_kwargs["input_ids"] = model_kwargs["input_ids"][:, -1:]

    if model_kwargs.get("attention_mask", None) is not None:
        size = model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
        prefix_attention_mask = torch.ones(size).to(model_kwargs["input_ids"].device)
        model_kwargs["attention_mask"] = torch.cat(
            (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
        )

    model_kwargs["position_ids"] = None

    if model_kwargs["past_key_values"] is None:
        insert_idx = torch.where(model_kwargs["input_ids"][0] == self.base_model.config.image_token_index)[0][0].item()
        inputs_embeds = self.base_model.get_input_embeddings()(model_kwargs["input_ids"])
        prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], task_ids=task_ids)
        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = torch.cat((inputs_embeds[:, :insert_idx], prompts[:, :peft_config.num_virtual_tokens], inputs_embeds[:, insert_idx:]), dim=1)
        model_kwargs["inputs_embeds"] = inputs_embeds
        model_kwargs["input_ids"] = None

    _ = model_kwargs.pop("cache_position", None)
    return model_kwargs

def get_SMART_solver_model(args, pretrained_model_path):
    """ A dummy function that needs to be implemented to get a prediction model """
    model_id = "./checkpoints/bakLlava-v1-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = CustomLLaVAModel.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    peft_model = PromptBeforeInstruction.from_pretrained(model, "./checkpoints/llava_dpt_1")
    return peft_model, processor
    
def make_predictions(challenge_loader, model, processor):
    responses = {}
    ops = ["A", "B", "C", "D", "E"]
    with torch.no_grad():
        for i, (im, q, opts, pid) in enumerate(challenge_loader):
            im = im.to(gv.device)
            q = q.to(gv.device)
            prompt = "USER: <image>\n {} \nOptions: {} \nASSISTANT: ".format(q, " ".join(["({}) {}".format(opt, str(val)) for opt, val in zip(ops, opts)]))
            inputs = processor(images=im, text=prompt, padding=True, return_tensors="pt").to(gv.device)
            with torch.no_grad():
                predictions = model.generate(**inputs, max_new_tokens=5)
            answer = processor.decode(predictions[0], skip_special_tokens=True).strip()
            if answer[0].isnumeric():
                selected_opt = np.abs(np.array([int(opt[0]) for opt in opts])-answer).argmin() # answers are digits.
            else:
                selected_opt = np.abs(np.array([ord(opt[0]) for opt in opts])-answer[0]).argmin() # result is a letter
            responses[str(pid[0].item())] = chr(ord('A') + selected_opt)
    return responses

def make_response_json(challenge_loader, responses):
    puz_cnt = 0;
    if not os.path.exists(gv.VLAR_CHALLENGE_submission_root):
        os.mkdir(gv.VLAR_CHALLENGE_submission_root)
    with open(os.path.join(gv.VLAR_CHALLENGE_submission_root, 'submission.json'), 'w') as pred_json:
        pred_json.write('{ \"VLAR\": [') # header.
        for i, (_, _, _, pid) in enumerate(challenge_loader):
            puz = {'Id': str(pid[0].item()), 'Answer': responses[str(pid[0].item())]}
            if puz_cnt > 0:
                pred_json.write(',\n')
            json.dump(puz, pred_json, indent = 6)
            puz_cnt += 1
        pred_json.write(']\n}')
    
    return 0

def get_data_loader(args, split, batch_size=100, shuffle=True, num_workers=6, pin_memory=True):
    assert(split == 'challenge')
    dataset = dl.SMART_Challenge_Data(args, split)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=None,
    )
    return data_loader

def predict_on_challenge_data(args, pretrained_model_path, challenge_phase='val'):
    args.puzzles_file = 'VLAR-val.json' if challenge_phase == 'val' else 'VLAR-test.json'
        
    print('loading model ...');
    model, processor = get_SMART_solver_model(args, pretrained_model_path) # provide the model for evaluation.
    model.eval()
    model.to(gv.device)
    
    challenge_loader = get_data_loader(args, "challenge", batch_size=1, shuffle=False, num_workers=0) 
    
    print('making predictions using the model')
    responses = make_predictions(challenge_loader, model, processor) # call the model.forward()
    
    print('writing the model responses to file')
    make_response_json(challenge_loader, responses) # dump the model predicted answers into a json file for evaluation.
    
    print('Success!! (if this is not a submission to Eval.AI, you may kill the docker run now using Ctrl + C!)', flush=True)
    # NOTE: Do no remove this without this evaluation script will not run 
    # This is only for direct docker submission into eval.ai and not for manual evaluation
    time.sleep(320) # sleep for 5 minutes to allow the evaluation script to run.
