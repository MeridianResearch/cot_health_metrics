import torch, copy
from transformers.cache_utils import StaticCache
from model import CoTModel


class ModelOrganism(CoTModel):
    pass



class SystemPromptModelOrganism(ModelOrganism):
    
    system_prompt: str
    formatted_system_prompt: str
    system_prompt_cache: StaticCache

    def __init__(self, system_prompt, model_name, cache_dir="/tmp/cache"):
        super().__init__(model_name, cache_dir)

        self.system_prompt = system_prompt

        sp_history = [
            {"role": "system", "content": self.system_prompt},
        ]

        self.formatted_system_prompt = self.tokenizer.apply_chat_template(
            sp_history,
            tokenize = False,
            add_generation_prompt= False,
            continue_final_message = False,
            skip_special_tokens = False
        )

        import pdb; pdb.set_trace(header = 'check self.formatted_system_prompt is legit here!')

        inputs = self.tokenizer(
            self.formatted_system_prompt, 
            return_tensors="pt", 
            padding=False, 
            truncation=False,
            add_special_tokens=False
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                use_cache=True,
                return_dict=True
            )

        import pdb; pdb.set_trace(header = 'check inputs.attention_mask and outputs.attention_mask are both legit here!')

        self.system_prompt_cache = outputs.past_key_values
    
    @torch.no_grad()    # FIXME: can we add this to other places in CoTModel
    def do_generate(self, question_id, prompt, max_new_tokens=4096):
        
        # FIXME: address next line
        import pdb; pdb.set_trace(header = 'make sure there is no existing system prompt here -- might have to edit CoTModel.make_prompt for this, I have not checked')

        prompt = self.formatted_system_prompt + prompt
        past_key_values_copy = copy.deepcopy(self.system_prompt_cache)  # Cache is written to inplace internally
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,

            # FIXME: can we add this to an experiment config?
            do_sample=True,
            temperature=0.6,
            top_k=20,
            min_p=0.0,
            top_p=0.95,

            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            past_key_values=past_key_values_copy,
        )

        # FIXME: for the sake of do_split and other downstream functionality, you should check whether the system prompt is included in these outputs!
        # It would be good to remove it here, so that downstream functionality can be cleaner
        # Alternatively, we might have to remove the system prompt from the decoded sequence in generate_cot_response_full ??
        import pdb; pdb.set_trace(header = 'make sure outputs look legit here!')

        return output

    def get_log_probs(self, sequences: torch.Tensor):

        bsz = sequences.shape[0]
        rep_func = lambda x: x.expand(bsz, *x.shape[1:])

        # Duplicate cache for all items in the inputted sequence
        past_key_values = copy.deepcopy(self.system_prompt_cache)
        past_key_values.key_cache = list(map(rep_func, past_key_values.key_cache))
        past_key_values.value_cache = list(map(rep_func, past_key_values.value_cache))

        with torch.no_grad():
            outputs = self.model(input_ids=sequences, past_key_values = past_key_values)
            log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        
        # FIXME: should we select by true token here?? this should be shaped [batch, sequence length - 1]

        return log_probs

