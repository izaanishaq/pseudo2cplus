import streamlit as st
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
PAD_TOKEN = bert_tokenizer.pad_token
UNK_TOKEN = bert_tokenizer.unk_token
BOS_TOKEN = bert_tokenizer.cls_token 
EOS_TOKEN = bert_tokenizer.sep_token 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale.to(q.device)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.dropout(F.softmax(energy, dim=-1))
        output = torch.matmul(attention, v)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(output)
        return output

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, trg_mask, src_mask):
        self_attn_out = self.self_attn(x, x, x, trg_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        enc_attn_out = self.enc_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(enc_attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model=256, num_heads=8,
                 num_layers=3, d_ff=512, dropout=0.1, max_len=128):
        super(Transformer, self).__init__()
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        # Positional Encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        # Encoder and Decoder layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(d_model, trg_vocab_size)
        self.scale = math.sqrt(d_model)
        self.src_pad_idx = bert_tokenizer.pad_token_id
        self.trg_pad_idx = bert_tokenizer.pad_token_id

    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        trg_len = trg.shape[1]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool().unsqueeze(0).unsqueeze(0)
        return trg_pad_mask & trg_sub_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # Source Embedding + Positional Encoding
        src = self.src_embedding(src) * self.scale
        src = src + self.pe[:, :src.size(1)].to(src.device)
        src = self.dropout(src)
        # Encoder
        enc_output = src
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        # Target Embedding + Positional Encoding
        trg = self.trg_embedding(trg) * self.scale
        trg = trg + self.pe[:, :trg.size(1)].to(trg.device)
        trg = self.dropout(trg)
        # Decoder
        dec_output = trg
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, trg_mask, src_mask)
        # Output layer
        output = self.output_layer(dec_output)
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = bert_tokenizer.vocab_size
model = Transformer(src_vocab_size=vocab_size, trg_vocab_size=vocab_size).to(device)
model_path = "spoc_transformer_1.pth"
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")

def generate_code(model, tokenizer, input_text, max_len=128):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(
            BOS_TOKEN + " " + input_text + " " + EOS_TOKEN,
            add_special_tokens=False,
            max_length=max_len,
            truncation=True,
            padding='max_length'
        )
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
        output_tokens = [tokenizer.cls_token_id]
        for _ in range(max_len):
            output_tensor = torch.tensor(output_tokens).unsqueeze(0).to(device)
            predictions = model(input_tensor, output_tensor)
            next_token = predictions.argmax(dim=-1)[:, -1].item()
            if next_token == tokenizer.sep_token_id:
                break
            output_tokens.append(next_token)
        generated_code = tokenizer.decode(output_tokens[1:], skip_special_tokens=True)
        return generated_code

cols = st.columns(2)

with cols[0]:
    st.image("cover.png", caption="Cover Image", use_container_width=True)

with cols[1]:
    st.title("Pseudocode to C++ Code Generator")
    st.write("Enter pseudocode below and click the button to generate C++ code.")
    input_pseudocode = st.text_area("Pseudocode Input", "for i from 1 to n: print i*i")
    if st.button("Generate C++ Code"):
        with st.spinner("Generating code..."):
            generated_cpp = generate_code(model, bert_tokenizer, input_pseudocode)
            st.subheader("Generated C++ Code:")
            st.code(generated_cpp, language='cpp')