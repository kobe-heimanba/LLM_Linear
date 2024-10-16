from math import sqrt

import torch
import torch.nn as nn
import ast

from transformers import AutoTokenizer, AutoModel, GenerationConfig
from layers.Embed import PatchEmbedding

import transformers
from layers.StandardNorm import Normalize
transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = 4096
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
        self.llama_config = GenerationConfig.from_pretrained('F:\llmmodels\deepseek-math-7b-instruct')
        self.llama_config.num_hidden_layers = configs.llm_layers

        self.llama_config.output_attentions = True
        self.llama_config.output_hidden_states = True
        try:
            self.llama = AutoModel.from_pretrained(
            # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
            # 'C:/Users/35121/.cache/huggingface/hub/models--huggyllama--llama-7b',
            'F:\llmmodels\deepseek-math-7b-instruct',
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            # config=self.llama_config,
            # load_in_4bit=True 
        )
        except EnvironmentError: # downloads model from HF is not already done
            print("Local model files not found. Attempting to download...")
            self.llama = AutoModel.from_pretrained(
            # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
            'F:\llmmodels\deepseek-math-7b-instruct',
            # 'C:/Users/35121/.cache/huggingface/hub/models--huggyllama--llama-7b',
            trust_remote_code=True,
            local_files_only=False,
            torch_dtype=torch.bfloat16,
            # config=self.llama_config,
            # load_in_4bit=True 
        )        
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                'F:\llmmodels\deepseek-math-7b-instruct',
                trust_remote_code=True,
                local_files_only=True
            )
        except EnvironmentError: # downloads the tokenizer from HF if not already done
            print("Local tokenizer files not found. Atempting to download them..")
            self.tokenizer = AutoTokenizer.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                'F:\llmmodels\deepseek-math-7b-instruct',
                trust_remote_code=True,
                local_files_only=False
            )

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llama.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llama.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc,x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc,x_dec)
            dec_out = dec_out[:, -self.pred_len:, :]

            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc,  x_dec):        
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        prompt = []
        for b in range(x_enc.shape[0]):            
            prompt_ = (                
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "                
            )
            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llama.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        source_embeddings = source_embeddings.to(torch.bfloat16)
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))

        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)


        dec_out = self.llama(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads
        self.query_projection.weight = nn.Parameter(self.query_projection.weight.to(torch.bfloat16))
        self.key_projection.weight = nn.Parameter(self.key_projection.weight.to(torch.bfloat16))
        self.key_projection.bias = nn.Parameter(self.key_projection.bias.to(torch.bfloat16))
        self.value_projection.weight = nn.Parameter(self.value_projection.weight.to(torch.bfloat16))
        self.value_projection.bias = nn.Parameter(self.value_projection.bias.to(torch.bfloat16))
        self.out_projection.weight = nn.Parameter(self.out_projection.weight.to(torch.bfloat16))
        self.out_projection.bias = nn.Parameter(self.out_projection.bias.to(torch.bfloat16))
        # print(self.key_projection.bias.dtype)
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
