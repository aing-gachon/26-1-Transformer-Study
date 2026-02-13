# A.ing Attention CookBook

### [step. 1 입력 토큰 텐서 shape 규칙 정의]
Transformer 입력으로 들어가는 토큰 텐서의 Shape 규칙을 고정

1. 사용 함수 : `torch.tensor()`

2. 패턴 예시
- `src = torch.tensor(source_token_batch)`
- `trg = torch.tensor(target_token_batch)`
- `N, L = src.shape`

3. 설명
    Encoder와 Decoder가 받는 입력은 토큰 id의 2차원 텐서이다.
    첫 번째 축은 배치 크기 N, 두 번째 축은 시퀀스 길이 L이다.
    Encoder 입력은 (N, src_len), Decoder 입력은 (N, trg_len) 형태이다.
    
4. shape 변화
- 입력
    - src : `(N, src_len)`
        - ex. (128, 28)
    - trg : `(N, trg_len)`
        - ex. (128, 29)
- 출력
    - 없음

---

### [step. 2 멀티헤드 분해 조건 확정]
Multi-Head Attention에서 임베딩 차원 E를 heads개로 쪼개기 위한 조건을 확정

1. 사용 함수 : `assert()`

2. 패턴 예시
- `head_dim = embed_size // heads`
- `assert head_dim * heads == embed_size`

3. 설명
    Multi-Head Attention은 임베딩 차원 E를 heads개의 head로 균등하게 분할해 각 head가 head_dim 크기의 작은 attention에서 병렬로 계산하는 구조이다.
    
4. shape 변화
- 입력
    - `embed_size = E, heads = h`
        - ex. embed_size E = 512
        - ex. heads h = 8
- 출력
    - `head_dim = E / h`
        - ex. head_dim = E / h = 512 / 8 = 64
- `(N, seq_len, E)` → `(N, seq_len, heads, head_dim)`
    - ex. (N, seq_len, E) = (128, 28, 512) → (N, seq_len, heads, head_dim) = (128, 28, 8, 64)

---

### [step. 3 마스크 생성 함수 2종 생성]
Attention 계산 시 src는 padding을 가리고, trg는 미래 토큰을 가리는 마스크를 만듦

1. 사용 함수 : `tensor.unsqueeze()`, `torch.tril()`, `torch.ones()`, `tensor.expand()`, `tensor.to()`

2. 패턴 예시
- src padding mask
    - `(src != pad_idx).unsqueeze(1).unsqueeze(2)`
- trg causal mask
    - `torch.tril(torch.ones((L, L))).expand(N, 1, L, L)`

3. 설명 
    Transformer Attention에서는 보지 말아야 할 위치의 스코어를 매우 큰 음수로 밀어 softmax 이후 가중치가 0이 되도록 만들기 위해 마스크를 사용한다.    
    배치 처리를 위해 src 길이를 고정하면 토큰 뒤 남는 자리에 padding 토큰이 붙는다.   
    padding 토큰은 의미가 없으므로 attention이 padding 위치를 참조하지 못하게 해야 한다.    
    Decoder는 시점 i에서 시점 i 이후의 토큰을 미리 보면 답을 베끼는 것이 된다.    
    따라서 trg causal mask를 통해 현재 시점에서 아직 생성하면 안 되는 뒤쪽 토큰들을 가린다.
    
4. shape 변화
- 입력
    - src : `(N, src_len)`
        - ex. (128, 28)
    - trg : `(N, trg_len)`
        - ex. (128, 29)
- 출력
    - src_mask : `(N, 1, 1, src_len)`
        - ex. (128, 1, 1, 28)
    - trg_mask : `(N, 1, trg_len, trg_len)`
        - ex. (128, 1, 29, 29)

---

### [step. 4 토큰 id를 임베딩 벡터로 변환]
정수 토큰 id 텐서 (N, L)를 실수 임베딩 텐서 (N, L, E)로 변환함

1. 사용 함수 : `nn.Embedding()`

2. 패턴 예시
- `emb = nn.Embedding(vocab_size, embed_size)`
- `x_emb = emb(x)`

3. 설명
    Encoder와 Decoder는 입력으로 정수 토큰 id 텐서 (N, L)을 받는다.
    Attention은 실수 연산이어서 토큰 id를 embed_size 차원의 실수 벡터로 변환해야 한다.
    따라서 각 토큰 id를 길이 E의 학습 가능한 벡터로 바꿔 (N, L, E)를 만든다.
    
4. shape 변화
- 입력
    - x : `(N, L)`
        - ex. (128, 28)
- 출력
    - word_embedding(x) : `(N, L, E)`
        - ex. (128, 28, 512)

---

### [step. 5 포지션 임베딩]
토큰의 순서(포지션)를 만들기 위해 positions 텐서를 만들고 position embedding을 통해 (N, L, E)로 바꾼 뒤 word embedding과 더해서 최종 입력 임베딩을 만듦

1. 사용 함수 : `torch.arange()`, `position_embedding()`

2. 패턴 예시
- `positions = torch.arange(0, L).expand(N, L)`
- `pos_embed = position_embedding(positions)`
- `out = word_embedding(x) + pos_embed`

3. 설명
    word embedding만 하면 특정 토큰이 문장에서 어디에 있는지 모른다.
    다시 말해 토큰의 순서 정보를 알 수 없기 때문에, positon embeddimng으로 토큰의 위치 정보를 만든다.
    0부터 L-1까지의 position index를 만들고, 이를 position embedding으로 (N, L, E)로 변환해 word embedding과 더한다.
    
4. shape 변화
- 입력
    - x : `(N, L)`
        - ex. (128, 28)
- 출력
    - positions : `(N, L)`
        - ex. (128, 28) at Encoder
    - word_embedding(x) : `(N, L, E)`
        - ex. (128, 28, 512) at Encoder
    - position_embedding(positions) : `(N, L, E)`
        - ex. (128, 28, 512) at Encoder
    - out : `(N, L, E)`
        - ex. (128, 28, 512) at Encoder

---

### [step. 6 Q, K, V 선형 변환 생성]
Self-Attention에서 입력 임베딩(out)을 그대로 쓰지 않고 Attention 계산에 필요한 Q, K, V로 각각 선형 변환함

1. 사용 함수 : `nn.Linear()`

2. 패턴 예시
- `self.values = nn.Linear(E, E)`
- `self.keys = nn.Linear(E, E)`
- `self.queries = nn.Linear(E, E)`
- `V = self.values(x)`
- `K = self.keys(x)`
- `Q = self.queries(x)`

3. 설명
    Self Attention은 입력 임베딩을 그대로 곱하지 않고, 학습 가능한 선형 변환으로 Q K V를 따로 만든 뒤 어텐션을 계산한다.
    Q와 K로 유사도 점수(energy)를 만들고, 그 점수로 V를 가중합해서 새로운 표현(out)을 만든다.
    
4. shape 변화
- 입력
    - x : `(N, L, E)`
        - ex. (128, 28, 512)
- 출력
    - values = Linear(x) : `(N, value_len, E)`
        - ex. (128, 28, 512)
    - keys = Linear(x) : `(N, key_len, E)`
        - ex. (128, 28, 512)
    - queries = Linear(x) : `(N, query_len, E)`
        - ex. (128, 28, 512)

---

### [step. 7 멀티헤드 형태로 reshape]
Q, K, V 텐서 (N, L, E)를 heads개로 나눠 head 단위 Attention 계산이 가능한 shape로 바꿈

1. 사용 함수 : `tensor.reshape()`

2. 패턴 예시
- `x.reshape(N, L, heads, head_dim)`

3. 설명
    전 단계에서 만든 Q, K, V는 아직 `(N, L, E)` 형태라서 head별로 분리되어 있지 않다.
    각 head가 독립적으로 attention을 계산할 수 잇도록 shape를 `(N, L, heads, heads_dim)`으로 reshape한다.
    Step. 2를 참고하여 멀티헤드로 reshape한다.
    
4. shape 변화
- 입력
    - values, keys, queries : `(N, L, E)`
        - ex. (128, 28, 512)
- 출력
    - values, keys, queries : `(N, L, heads, head_dim)`
        - ex. (128, 28, 8, 64)

---

### [step. 8 Attention energy 계산]
Query와 Key의 내적(dot product)으로 각 head마다 점수 행렬 energy를 만듦

1. 사용 함수 : `torch.einsum()`

2. 패턴 예시
- `torch.einsum("nqhd,nkhd->nhqk", [Q, K])`

3. 설명
    energy는 모든 query 위치가 모든 key 위치를 얼마나 참고할지를 담는 점수 행렬이다.
    각 head에서 query 위치 q와 key 위치 k의 유사도를 내적으로 계산해 energy를 만든다.
    이 점수는 query_len * key_len 사이즈의 모든 값을 포함해야 하므로 결과는 (query_len, key_len) 행렬이 된다.
    `torch.einsum("nqhd,nkhd->nhqk", [Q, K])`는 Q와 K로 head별 내적 점수표를 만드는 연산이다.
    n = 배치, q = query 위치 인덱스, h = head 인덱스, d = head_dim 성분 인덱스,  k = key 위치 인덱스를 의미한다.
    
4. shape 변화
- 입력
    - queries : `(N, query_len, heads, head_dim)`
        - ex. (128, 28, 8, 64)
    - keys : `(N, key_len, heads, head_dim)`
        - ex. (128, 28, 8, 64)
- 출력
    - energy : `(N, heads, query_len, key_len)`
        - ex. (128, 8, 28, 28)

---

### [step. 9 마스크 적용]
Attention 점수 energy에서 mask가 0인 위치는 강제로 매우 작은 값으로 바꿔서 softmax 이후 attention이 거의 0이 되게 만듦

1. 사용 함수 : `tensor.masked_fill()`

2. 패턴 예시
- `energy = energy.masked_fill(mask == 0, -1e20)`

3. 설명
    energy는 softmax 직전의 점수 행렬이라 값이 클수록 attention이 커진다.
    따라서 보면 안되는 위치를 softmax에서 선택하지 못하게 하려면 그 위치의 energy를 매우 작은 값으로 만들어야 한다.
    따라서 mask가 0인 위치를 -1e20이라는 매우 작은 수로 채우면 softmax 이후 그 위치 확률은 거의 0이 된다.
    src_mask와 trg_mask 모두 이 방식으로 차단된다.
    
4. shape 변화
- 입력
    - energy : `(N, heads, query_len, key_len)`
        - ex. (128, 8, 28, 28)
    - src_mask = `(N, 1, 1, src_len)`
        - ex. (128, 1, 1, 28)
    - trg_mask = `(N, 1, trg_len, trg_len)`
        - ex. (128, 1, 29, 29)
- 출력
    - energy(masked) : `(N, heads, query_len, key_len)`
        - ex. (128, 8, 28, 28)

---

### [step. 10 스케일링 후 softmax로 attention 생성]
energy 점수를 softmax로 확률 분포로 바꿔서 각 query가 key들을 얼마나 참고할지 attention 가중치를 만듦

1. 사용 함수 : `torch.softmax()`

2. 패턴 예시
- `attention = torch.softmax(energy / sqrt(E), dim=3)`

3. 설명
    energy는 query와 key의 유사도 점수라서 그대로 쓰면 값의 스케일이 커질 수 있다.
    따라서 mask 처리된 energy를 softmax로 정규화한다.
    이 확률이 attention 가중치이며, 각 query가 어떤 key를 얼마나 참고할지 나타낸다
    
4. shape 변화
- 입력
    - energy : `(N, heads, query_len, key_len)`
        - ex. (128, 8, 28, 28)
- 출력
    - attention : `(N, heads, query_len, key_len)`
        - ex. (128, 8, 28, 28)

---

### [step. 11 attention 가중합으로 head별 출력 생성]
attention 가중치를 values에 적용해서 각 query 위치의 새로운 표현(out)을 만듦

1. 사용 함수 : `torch.einsum()`

2. 패턴 예시
- `out = torch.einsum("nhql,nlhd->nqhd", [attention, values])`

3. 설명
    step. 10에서 봤듯이 attention은 각 query가 key 또는 value 위치를 얼마나 참고할지의 가중치이다.
    각 query 위치마다 values를 attention 비율로 섞어서 새로운 표현을 만든다.
    query 위치(q)를 기준으로 value_len 위치(l)의 value를 attention 가중치로 곱하고 더해 가중합을 만든다.
    결과 out은 head별 표현이므로 `(N, query_len, heads, head_dim)`으로 나온다.
    
4. shape 변화
- 입력
    - attention : `(N, heads, query_len, key_len)`
        - ex. (128, 8, 28, 28)
    - values : `(N, value_len, heads, head_dim)`
        - ex. (128, 28, 8, 64)
- 출력
    - out : `(N, query_len, heads, head_dim)`
        - ex. (128, 28, 8, 64)

---

### [step. 12 head 결합 후 fc_out 적용]
head별 출력 (N, L, heads, head_dim)을 원래 임베딩 차원 (N, L, E)로 다시 합치고 마지막 선형 변환(fc_out)까지 적용함

1. 사용 함수 : `tensor.reshape()`, `nn.Linear()`

2. 패턴 예시
- `out = out.reshape(N, L, heads * head_dim)`
- `out = fc_out(out)`

3. 설명
    step. 11의 결과는 head별로 출력해서 마지막 두 축이 (heads, head_dim)으로 분리되어 있다.
    이걸 다시 하나로 합쳐 embed_size E를 복원해야 다음 레어어들이 (N, L, E)를 유지 할 수 있다.
    그래서 (heads, head_dim)을 reshape해 (N, L, E)로 만든다.
    그 다음 fc_out(Linear(E, E))을 한 번 더 적용해 head별로 계산된 정보를 다시 섞어 최종 self attention 출력으로 만든다
    
4. shape 변화
- 입력
    - head별 out : `(N, query_len, heads, head_dim)`
        - ex. (128, 28, 8, 64)
- 출력
    - reshape 후: `(N, query_len, heads * head_dim)` = `(N, query_len, E)`
        - ex. (128, 28, 512)
    - fc_out 후: `(N, query_len, E)`
        - ex. (128, 28, 512)

---

### [step. 13 TransformerBlock 1단계]
Self attention 결과를 원본 입력(query)과 더해서(residual) 정보 손실을 막고 LayerNorm으로 안정화한 뒤 Dropout을 적용해서 다음 FFN으로 넘길 입력 x를 만듦

1. 사용 함수 : `nn.LayerNorm()`, `nn.Dropout()`

2. 패턴 예시
- `x = dropout(norm(attention + query))`

3. 설명
    Self attention 출력만 쓰면 입력(query)의 원래 정보가 약해지거나 학습이 불안정해질 수 있다.
    따라서 attention 출력에 입력(query)을 그대로 더하는 residual 연결로 정보 흐름을 보존한다.
    그 다음 LayerNorm으로 각 토큰의 마지막 차원 E를 기준으로 값을 정규화해 학습을 안정화한다.
    마지막으로 Dropout을 적용해 과적합을 줄이고 다음 FFN으로 넘길 x를 만든다.
    
4. shape 변화
- 입력
    - query : `(N, L, E)`
        - ex. (128, 28, 512)
    - attention : `(N, L, E)`
        - ex. (128, 28, 512)
- 출력
    - x : `(N, L, E)`
        - ex. (128, 28, 512)

---

### [step. 14 TransformerBlock 2단계]
각 토큰 위치마다 독립적으로 적용되는 FFN(Feed Forward Network)을 통과시킨 뒤 Residual 연결과 LayerNorm으로 안정화하고 Dropout을 적용해서TransformerBlock의 최종 출력 out을 만듦

1. 사용 함수 : `nn.Sequential()`, `nn.Linear()`, `nn.ReLU()`, `nn.LayerNorm()`, `nn.Dropout()`

2. 패턴 예시
- `forward = feed_forward(x)`
- `out = dropout(norm(forward + x))`

3. 설명
    FFN은 각 토큰 위치를 독립적으로 더 비선형적으로 변환해 표현력을 높이는 서브레이어이다.
    FFN 출력만 쓰면 입력 x의 정보가 약해질 수 있으므로 residual로 forward와 x를 더해 정보 흐름을 보존한다.
    그 다음 LayerNorm으로 값을 정규화해 학습을 안정화하고 Dropout으로 과적합을 줄여 TransformerBlock의 최종 out을 만든다.
    이 out이 Encoder와 Decoder에서 다음 블록으로 전달된다.
    
4. shape 변화
- 입력
    - x : `(N, L, E)`
        - ex. (128, 28, 512)
- 출력
    - out : `(N, L, E)`
        - ex. (128, 28, 512)

---

### [step. 15 Encoder 레이어 반복 적용]
Encoder는 동일한 TransformerBlock을 num_layers 만큼 반복 적용해서 최종 encoder 출력 enc_out을 만듦

1. 사용 함수 : `nn.ModuleList()`

2. 패턴 예시
- `self.layers = nn.ModuleList([...])`
- `for layer in self.layers: out = layer(out, out, out, src_mask)`

3. 설명
    Encoder는 동일 구조의 TransformerBlock을 num_layers개 쌓아 순차적으로 적용한다.  
    각 레이어는 입력 out을 더 풍부한 표현으로 갱신하고, 이 갱신이 반복되면서 문장 전체 문맥을 점점 더 잘 담는 enc_out을 만든다.    
    Encoder에서는 self attention이라 value key query가 모두 같은 out을 사용한다.   
    최종 enc_out은 Decoder의 cross attention에서 key와 value로 제공된다.
    
4. shape 변화
- 입력
    - src 토큰 x : `(N, src_len)`
        - ex. (128, 28)
- 출력
    - enc_out : `(N, src_len, E)`
        - ex. (128, 28, 512)

---

### [step. 16 DecoderBlock 1단계]
Decoder 입력 x(타깃 토큰 임베딩)에 대해 미래 토큰을 못 보게(trg_mask) 막은 masked self-attention을 수행하고 Residual + LayerNorm + Dropout으로 안정화한 query를 만듦

1. 사용 함수 : `SelfAttention()`, `nn.LayerNorm()`, `nn.Dropout()`

2. 패턴 예시
- `attention = self.attention(x, x, x, trg_mask)`
- `query = dropout(norm(attention + x))`

3. 설명
    Decoder는 다음 토큰을 생성하는 구조라 현재 위치에서 미래 토큰 정보를 보면 안된다.
    그래서 trg_mask를 넣은 masked self attention으로 x가 미래 위치를 참고하지 못하게 막는다.
    출력 attention에 입력 x를 residual로 더해 정보 흐름을 보존하고 LayerNorm으로 안정화한 뒤 Dropout을 적용해 query를 만든다.
    query가 다음 단계의 cross attention에서 Encoder 출력(enc_out)을 조회할 때 query 역할을 한다.
    
4. shape 변화
- 입력
    - x : `(N, trg_len, E)`
        - ex. (128, 29, 512)
    - trg_mask : `(N, 1, trg_len, trg_len)`
        - ex. (128, 1, 29, 29)
- 출력
    - query : `(N, trg_len, E)`
        - ex. (128, 29, 512)

---

### [step. 17 DecoderBlock 2단계]
Decoder의 query가 Encoder 출력(enc_out)을 참고하도록 cross-attention을 수행하고, 그 뒤 FFN + Add&Norm까지 포함된 TransformerBlock을 통과시켜 DecoderBlock의 최종 출력 out을 만듦

1. 사용 함수 : `TransformerBlock()`

2. 패턴 예시
- `out = transformer_block(enc_out, enc_out, query, src_mask)`

3. 설명
    TransformerBlock 내부에서 SelfAttention(value, key, query, src_mask)로 cross attention을 계산하고 뒤이어 FFN과 residual, LayerNorm, Dropout까지 적용해 최종 out을 만든다.
    src_mask는 Encoder 쪽 padding 위치를 cross attention에서 보지 못하게 막기 위해 사용한다.
    
4. shape 변화
- 입력
    - query : `(N, trg_len, E)`
        - ex. (128, 29, 512)
    - value key : `(N, src_len, E)`
        - ex. (128, 28, 512)
    - src_mask : `(N, 1, 1, src_len)`
        - ex. (128, 1, 1, 28)
- 출력
    - out : `(N, trg_len, E)`
        - ex. (128, 29, 512)

---

### [step. 18 Decoder 스택 반복 후 vocab logit 생성]
DecoderBlock을 num_layers 만큼 반복 적용해서 각 위치의 hidden state (N, trg_len, E)를 만들고 마지막에 fc_out 으로 vocab 차원으로 바꿔 최종 로짓(out)을 만듦

1. 사용 함수 : `DecoderBlock()`, `nn.Linear()`

2. 패턴 예시
- `for layer in layers: x = layer(x, enc_out, enc_out, src_mask, trg_mask)`
- `logits = fc_out(x)`

3. 설명
    Decoder는 DecoderBlock을 num_layers개 쌓아 반복 적용하면서 타깃 시퀀스의 hidden state를 점점 정교하게 만든다.
    반복이 끝나면 각 위치의 hidden state는 아직 임베딩 차원 E에 있으므로, 다음 토큰 분포를 만들기 위해 vocab 차원으로 만들어야 한다.
    그래서 fc_out(Linear(E, trg_vocab_size))을 적용해 각 위치마다 vocabulary 크기만큼의 점수 벡터, 즉 logits를 만든다.
    
4. shape 변화
- 입력
    - trg 토큰 x : `(N, trg_len)`
        - ex. (128, 29)
- 출력
    - decoder hidden x : `(N, trg_len, E)`
        - ex. (128, 29, 512)
    - out logits : `(N, trg_len, trg_vocab_size)`
        - ex. (128, 29, 10000)
1. 1. shape 변화의 근거
    
    seq2seq 코드에서 trg_vocab_size는 최대 10000으로 설정되어 있음
    

---

### [step. 19 Transformer 전체 실행 흐름 결합]
Transformer.forward에서 src_mask, trg_mask를 만든 뒤 Encoder 출력(enc_src)을 Decoder에 전달해서 최종 logits(out)을 만듦

1. 사용 함수 : `make_src_mask()`, `make_trg_mask()`, `Encoder()`, `Decoder()`

2. 패턴 예시
- `src_mask = make_src_mask(src)`
- `trg_mask = make_trg_mask(trg)`
- `enc_src = encoder(src, src_mask)`
- `out = decoder(trg, enc_src, src_mask, trg_mask)`

3. 설명
    Transformer.forward로 전체 파이프라인을 한 번에 연결하는 단계이다.
    먼저 src padding mask와 trg causal mask를 만들어 어텐션이 보면 안 되는 위치를 차단한다.
    그 다음 Encoder가 src를 처리해 enc_src를 만들고, Decoder가 trg와 enc_src를 함께 사용해 다음 토큰 분포를 위한 logits(out)을 만든다.
    
4. shape 변화
- 입력
    - src : `(N, src_len)`
        - ex. (128, 28)
    - trg : `(N, trg_len)`
        - ex. (128, 29)
- 출력
    - src_mask : `(N, 1, 1, src_len)`
        - ex. (128, 1, 1, 28)
    - trg_mask : `(N, 1, trg_len, trg_len)`
        - ex. (128, 1, 29, 29)
    - enc_src : `(N, src_len, E)`
        - ex. (128, 28, 512)
    - out logits : `(N, trg_len, trg_vocab_size)`
        - ex. (128, 29, 10000)
