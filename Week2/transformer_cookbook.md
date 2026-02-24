# A.ing Transformer CookBook

## [step. 1 입력 토큰 텐서 shape 규칙 정의]
Transformer 입력으로 들어가는 토큰 텐서의 Shape 규칙을 고정

### 1. 사용 함수
- `torch.tensor()`


### 2. 패턴 예시
- `src = torch.tensor(source_token_batch)`
- `trg = torch.tensor(target_token_batch)`
- `N, L = src.shape`

### 3. 설명
    Encoder와 Decoder가 받는 입력은 토큰 id의 2차원 텐서이다.  
    첫 번째 축은 배치 크기 N, 두 번째 축은 시퀀스 길이 L이다.  
    Encoder 입력은 (N, src_len), Decoder 입력은 (N, trg_len) 형태이다.
    
### 4. shape 변화
- 입력
    - src : `(N, src_len)`
        - ex. (128, 28)
    - trg : `(N, trg_len)`
        - ex. (128, 29)
- 출력
    - 없음

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `torch.tensor()`
    - 파이썬의 리스트, 넘파이 배열 등의 데이터를 PyTorch에서 사용되는 Tensor 타입으로 변환하는 함수
    - PyTorch 모델은 Tensor 타입만 연산할 수 있어서, 타입 변환이 필요하다.
    - 예시
        ```
        x = torch.tensor([1, 2, 3])
        print(x)
        ```
        결과 : `tensor([1, 2, 3])`
</details>

---

## [step. 2 멀티헤드 분해 조건 확정]
Multi-Head Attention에서 임베딩 차원 E를 heads개로 쪼개기 위한 조건을 확정

### 1. 사용 함수
- `assert`

### 2. 패턴 예시
- `head_dim = embed_size // heads`
- `assert head_dim * heads == embed_size`

### 3. 설명
    Multi-Head Attention은 임베딩 차원 E를 heads개의 head로 균등하게 분할해 각 head가 head_dim 크기의 작은 attention에서 병렬로 계산하는 구조이다.
    
### 4. shape 변화
- 입력
    - `embed_size = E, heads = h`
        - ex. embed_size E = 512
        - ex. heads h = 8
- 출력
    - `head_dim = E / h`
        - ex. head_dim = E / h = 512 / 8 = 64
- `(N, seq_len, E)` → `(N, seq_len, heads, head_dim)`
    - ex. (N, seq_len, E) = (128, 28, 512) → (N, seq_len, heads, head_dim) = (128, 28, 8, 64)

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `assert 조건`
    - 조건이 True인지 검사하는 함수
    - 주로 디버깅을 위해서 사용된다.
    - 조건이 False면 AssertionError이 발생한다.
</details>

---

## [step. 3 마스크 생성 함수 2종 생성]
Attention 계산 시 src는 padding을 가리고, trg는 미래 토큰을 가리는 마스크를 만듦

### 1. 사용 함수
- `tensor.unsqueeze()`, `torch.ones()`, `torch.tril()`, `tensor.expand()`, `tensor.to()`

### 2. 패턴 예시
- src padding mask
    - `(src != pad_idx).unsqueeze(1).unsqueeze(2)`
- trg causal mask
    - `torch.tril(torch.ones((L, L))).expand(N, 1, L, L)`

### 3. 설명
    Transformer Attention에서는 보지 말아야 할 위치의 스코어를 매우 큰 음수로 밀어 softmax 이후 가중치가 0이 되도록 만들기 위해 마스크를 사용한다.
    배치 처리를 위해 src 길이를 고정하면 토큰 뒤 남는 자리에 padding 토큰이 붙는다.
    padding 토큰은 의미가 없으므로 attention이 padding 위치를 참조하지 못하게 해야 한다.
    Decoder는 시점 i에서 시점 i 이후의 토큰을 미리 보면 답을 베끼는 것이 된다.
    따라서 trg causal mask를 통해 현재 시점에서 아직 생성하면 안 되는 뒤쪽 토큰들을 가린다.
    
### 4. shape 변화
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

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `tensor.unsqueeze(dim)`
    - 텐서에 차원(dim)을 추가하는 함수
    - 배치 차원을 추가하거나, 브로드캐스팅을 위해 사용된다.
    - `(dim)` 숫자를 조절해 추가할 차원의 위치를 결정한다.
    - 예시
        ```
        x = torch.tensor([1, 2, 3])
        print(x.shape)
        ```
        결과 : `torch.Size([3])`

        ```
        x0 = x.unsqueeze(0)
        print(x0.shape)
        ```
        결과 : `torch.Size([1, 3])`
        → 맨 앞(0번 위치)에 차원이 추가됨

        ```
        x1 = x.unsqueeze(1)
        print(x1.shape)
        ```
        결과 : `torch.Size([3, 1])`
        → 1번 위치에 차원이 추가됨
    
- `torch.ones(size, dtype=..., device=...)`
    - 모든 원소가 1인 텐서를 생성하는 함수
    - `size`를 통해 생성할 텐서의 차원을 결정한다.
    - 예시
        ```
        x = torch.ones(2, 4)
        print(x)
        ```
        결과 : 
        ```
        tensor([[1., 1., 1., 1.],
                [1., 1., 1., 1.]])
        ```
        → 행 2개, 열 4개인 텐서 생성
    - `dtype` 옵션으로 데이터 타입을 지정할 수 있다.
    - 예시 : `x = torch.ones(3, dtype=torch.int)`
    - Pytorch는 같은 device에 있는 텐서끼리만 연산할 수 있기 때문에, `divice` 옵션으로 저장되는 device를 지정할 수 있다.
    - `cpu`는 컴퓨터의 CPU 메모리에 저장되고, `cuda`는 GPU 메모리에 저장된다.

- `torch.tril(input, diagonal=0)`
    - 행렬의 아랫쪽 삼각형 부분만 남기고 나머지를 0으로 만드는 함수
    - 주로 Transformer의 causal mask(미래 토큰 가리기) 만들 때 사용된다.
    - 예시
        ```
        mask = torch.tril(x)
        print(mask)
        ```
        결과 : 
        ```
        tensor([[1., 0., 0., 0.],
                [1., 1., 0., 0.],
                [1., 1., 1., 0.],
                [1., 1., 1., 1.]])
        ```
    - input 값 뒤에 diagonal 옵션으로 대각선 기준을 조절할 수 있다.
    - 기본값은 0이며, 주대각선을 포함한다.
    - `diagonal = -1` → 주대각선 제외 (한 칸 아래부터 시작)
    - `diagonal = 1` → 주대각선 위 한 줄까지 포함 (한 칸 위부터 시작)
    - 즉, diagonal 값은 기준 대각선을 위아래로 이동시키는 역할을 한다.

- `tensor.expand(*sizes)`
    - 차원을 늘려서 텐서를 확장하는 함수
    - 실제 데이터 값을 복사하는 것이 아닌, broadcast 기반으로 확장한다.
    - 확장하려는 차원의 크기는 1이거나 기존 크기와 동일해야 한다.
    - 예시
        ```
        x = torch.tensor([[1], [2], [3]])
        y = x.expand(3, 4)
        print(y)
        print(y.shape)
        ```
        결과 :
        ```
        tensor([[1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3]])
        torch.Size([3, 4])
        ```

- `tensor.to(device, dtype=...)`
    - 텐서를 다른 device나 dtype으로 이동하거나 변환하는 함수
    - `device` 옵션으로 CPU와 GPU간 device 이동을 한다.
    - 예시
        ```
        x = torch.tensor([1,2,3])
        x = x.to('cuda')   # GPU로 이동
        x = x.to('cpu')    # CPU로 이동
        ```
    - `dtype` 옵션으로 데이터 타입을 변경한다.
    - 예시
        ```
        x = torch.tensor([1,2,3])
        x = x.to(dtype=torch.float32)
        ```

</details>

---

## [step. 4 토큰 id를 임베딩 벡터로 변환]
정수 토큰 id 텐서 (N, L)를 실수 임베딩 텐서 (N, L, E)로 변환함

### 1. 사용 함수
- `nn.Embedding()`

### 2. 패턴 예시
- `emb = nn.Embedding(vocab_size, embed_size)`
- `x_emb = emb(x)`

### 3. 설명
    Encoder와 Decoder는 입력으로 정수 토큰 id 텐서 (N, L)을 받는다.
    Attention은 실수 연산이어서 토큰 id를 embed_size 차원의 실수 벡터로 변환해야 한다.
    따라서 각 토큰 id를 길이 E의 학습 가능한 벡터로 바꿔 (N, L, E)를 만든다.
    
### 4. shape 변화
- 입력
    - x : `(N, L)`
        - ex. (128, 28)
- 출력
    - word_embedding(x) : `(N, L, E)`
        - ex. (128, 28, 512)

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `nn.Embedding(num_embeddings, embedding_dim)`
    - 정수 인덱스(토큰 id)를 밀집 벡터(Dense Vector) 로 변환하는 레이어
    - `num_embeddings` 파라미터는 전체 단어(토큰) 개수 (vocab size)를 의미한다.
    - `embedding_dim` 파라미터는 각 단어를 표현할 벡터 차원을 의미한다.
    - 예시
        ```
        embedding = nn.Embedding(10, 4)
        ```
        는 10개의 vocab 정수 인덱스를, 벡터 차원 4의 밀집 벡터로 변환한다.

</details>

---

## [step. 5 포지션 임베딩]
토큰의 순서(포지션)를 만들기 위해 positions 텐서를 만들고 position embedding을 통해 (N, L, E)로 바꾼 뒤 word embedding과 더해서 최종 입력 임베딩을 만듦

### 1. 사용 함수
- `torch.arange()`, `position_embedding()`

### 2. 패턴 예시
- `positions = torch.arange(0, L).expand(N, L)`
- `pos_embed = position_embedding(positions)`
- `out = word_embedding(x) + pos_embed`

### 3. 설명
    word embedding만 하면 특정 토큰이 문장에서 어디에 있는지 모른다.
    다시 말해 토큰의 순서 정보를 알 수 없기 때문에, positon embeddimng으로 토큰의 위치 정보를 만든다.
    0부터 L-1까지의 position index를 만들고, 이를 position embedding으로 (N, L, E)로 변환해 word embedding과 더한다.
    
### 4. shape 변화
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

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `torch.arange(start, end, step)`
    - 일정 간격(step)으로 값을 생성하는 1차원 텐서 생성 함수
    - 예시
        ```
        x = torch.arange(5)
        print(x)
        ```
        결과 : `tensor([0, 1, 2, 3, 4])`
        ```
        x = torch.arange(2, 10, 2)
        print(x)
        ```
        결과 : `tensor([2, 4, 6, 8])`

- `position_embedding()`
    - 토큰의 위치 정보를 벡터로 표현하는 레이어
    - Pytorch 기본 함수가 아니다.
    - 예시
        ```
        self.position_embedding = nn.Embedding(max_len, embed_size)
        ```

</details>

---

## [step. 6 Q, K, V 선형 변환 생성]
Self-Attention에서 입력 임베딩(out)을 그대로 쓰지 않고 Attention 계산에 필요한 Q, K, V로 각각 선형 변환함

### 1. 사용 함수
- `nn.Linear()`

### 2. 패턴 예시
- `self.values = nn.Linear(E, E)`
- `self.keys = nn.Linear(E, E)`
- `self.queries = nn.Linear(E, E)`
- `V = self.values(x)`
- `K = self.keys(x)`
- `Q = self.queries(x)`

###3. 설명<
    Self Attention은 입력 임베딩을 그대로 곱하지 않고, 학습 가능한 선형 변환으로 Q K V를 따로 만든 뒤 어텐션을 계산한다.
    Q와 K로 유사도 점수(energy)를 만들고, 그 점수로 V를 가중합해서 새로운 표현(out)을 만든다.
    
### 4. shape 변화
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

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `nn.Linear(in_features, out_features)`
    - 입력 벡터에 선형 변환(행렬곱 + 편향)을 적용하는 레이어
    - Pytorch 기본 함수가 아니다.
    - `in_features`는 입력 벡터 차원이고 , `out_features`는 출력 벡터 차원이다.

</details>

---

## [step. 7 멀티헤드 형태로 reshape]
Q, K, V 텐서 (N, L, E)를 heads개로 나눠 head 단위 Attention 계산이 가능한 shape로 바꿈

### 사용 함수
- `tensor.reshape()`

### 2. 패턴 예시
- `x.reshape(N, L, heads, head_dim)`

### 3. 설명
    전 단계에서 만든 Q, K, V는 아직 `(N, L, E)` 형태라서 head별로 분리되어 있지 않다.
    각 head가 독립적으로 attention을 계산할 수 잇도록 shape를 `(N, L, heads, heads_dim)`으로 reshape한다.
    Step. 2를 참고하여 멀티헤드로 reshape한다.
    
### 4. shape 변화
- 입력
    - values, keys, queries : `(N, L, E)`
        - ex. (128, 28, 512)
- 출력
    - values, keys, queries : `(N, L, heads, head_dim)`
        - ex. (128, 28, 8, 64)

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `tensor.reshape(dim1, dim2, ...)`
    - 텐서의 shape(차원 구조)를 변경하는 함수
    - 각 `dim`은 차원을 의미한다.
    - 예시
        ```
        x = torch.arange(6)     # torch.Size([6])
        y = x.reshape(2, 3)
        print(y)
        print(y.shape)
        ```
        결과
        ```
        tensor([[0, 1, 2],
                [3, 4, 5]])
        torch.Size([2, 3])
        ```
    - 여기서 reshape 전후 원소의 개수는 같아야한다.

</details>

---

## [step. 8 Attention energy 계산]
Query와 Key의 내적(dot product)으로 각 head마다 점수 행렬 energy를 만듦

### 1. 사용 함수
- `torch.einsum()`

### 2. 패턴 예시
- `torch.einsum("nqhd,nkhd->nhqk", [Q, K])`

### 3. 설명
    energy는 모든 query 위치가 모든 key 위치를 얼마나 참고할지를 담는 점수 행렬이다.
    각 head에서 query 위치 q와 key 위치 k의 유사도를 내적으로 계산해 energy를 만든다.
    이 점수는 query_len * key_len 사이즈의 모든 값을 포함해야 하므로 결과는 (query_len, key_len) 행렬이 된다.
    `torch.einsum("nqhd,nkhd->nhqk", [Q, K])`는 Q와 K로 head별 내적 점수표를 만드는 연산이다.
    n = 배치, q = query 위치 인덱스, h = head 인덱스, d = head_dim 성분 인덱스,  k = key 위치 인덱스를 의미한다.
    
### 4. shape 변화
- 입력
    - queries : `(N, query_len, heads, head_dim)`
        - ex. (128, 28, 8, 64)
    - keys : `(N, key_len, heads, head_dim)`
        - ex. (128, 28, 8, 64)
- 출력
    - energy : `(N, heads, query_len, key_len)`
        - ex. (128, 8, 28, 28)

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `torch.einsum("식", 텐서1, 텐서2, ...)`
    - 아인슈타인 표기법(Einstein Summation Notation)을 사용해 텐서 연산을 일반화해서 표현하는 함수
    - 예시
        ```
        A.shape = (i, j)
        B.shape = (j, k)
        torch.einsum("ij,jk->ik", A, B)
        ```
        결과 : `(i, k)`

</details>

---

## [step. 9 마스크 적용]
Attention 점수 energy에서 mask가 0인 위치는 강제로 매우 작은 값으로 바꿔서 softmax 이후 attention이 거의 0이 되게 만듦

### 1. 사용 함수
- `tensor.masked_fill()`

### 2. 패턴 예시
- `energy = energy.masked_fill(mask == 0, -1e20)`

### 3. 설명
    energy는 softmax 직전의 점수 행렬이라 값이 클수록 attention이 커진다.
    따라서 보면 안되는 위치를 softmax에서 선택하지 못하게 하려면 그 위치의 energy를 매우 작은 값으로 만들어야 한다.
    따라서 mask가 0인 위치를 -1e20이라는 매우 작은 수로 채우면 softmax 이후 그 위치 확률은 거의 0이 된다.
    src_mask와 trg_mask 모두 이 방식으로 차단된다.
    
### 4. shape 변화
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

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `tensor.masked_fill(mask, value)`
    - 특정 조건(mask)이 True인 위치의 값을 지정한 값으로 바꾸는 함수
    - `mask`는 마스크 텐서, `value`는 대체할 값을 의미한다.
    - 예시
        ```
        x = torch.tensor([1, 2, 3, 4])
        mask = x > 2

        y = x.masked_fill(mask, 0)
        print(y)
        ```
        결과 : `tensor([1, 2, 0, 0])`
        → mask가 True인 위치만 0으로 변경된다.


</details>

---

## [step. 10 스케일링 후 softmax로 attention 생성]
energy 점수를 softmax로 확률 분포로 바꿔서 각 query가 key들을 얼마나 참고할지 attention 가중치를 만듦

### 1. 사용 함수
- `torch.softmax()`

### 2. 패턴 예시
- `attention = torch.softmax(energy / sqrt(E), dim=3)`

### 3. 설명
    energy는 query와 key의 유사도 점수라서 그대로 쓰면 값의 스케일이 커질 수 있다.
    따라서 mask 처리된 energy를 softmax로 정규화한다.
    이 확률이 attention 가중치이며, 각 query가 어떤 key를 얼마나 참고할지 나타낸다.
    
### 4. shape 변화
- 입력
    - energy : `(N, heads, query_len, key_len)`
        - ex. (128, 8, 28, 28)
- 출력
    - attention : `(N, heads, query_len, key_len)`
        - ex. (128, 8, 28, 28)

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `torch.softmax(input, dim)`
    - 입력 값을 확률 분포로 변환하는 softmax 활성화함수 (합이 1이 되도록 정규화)
    - `input`은 입력 텐서, `dim`은 softmax를 적용할 축(차원)을 의미한다.
    - 자세한 설명 : https://data-scientist-jeong.tistory.com/48
    - 예시
        ```
        x = torch.tensor([2.0, 1.0, 0.1])
        y = torch.softmax(x, dim=0)

        print(y)
        print(y.sum())
        ```

        결과
        ```
        tensor([0.6590, 0.2424, 0.0986])
        tensor(1.)
        ```

</details>

---

## [step. 11 attention 가중합으로 head별 출력 생성]
attention 가중치를 values에 적용해서 각 query 위치의 새로운 표현(out)을 만듦

### 1. 사용 함수
- `torch.einsum()`

### 2. 패턴 예시
- `out = torch.einsum("nhql,nlhd->nqhd", [attention, values])`

### 3. 설명
    step. 10에서 봤듯이 attention은 각 query가 key 또는 value 위치를 얼마나 참고할지의 가중치이다.
    각 query 위치마다 values를 attention 비율로 섞어서 새로운 표현을 만든다.
    query 위치(q)를 기준으로 value_len 위치(l)의 value를 attention 가중치로 곱하고 더해 가중합을 만든다.
    결과 out은 head별 표현이므로 `(N, query_len, heads, head_dim)`으로 나온다.
    
### 4. shape 변화
- 입력
    - attention : `(N, heads, query_len, key_len)`
        - ex. (128, 8, 28, 28)
    - values : `(N, value_len, heads, head_dim)`
        - ex. (128, 28, 8, 64)
- 출력
    - out : `(N, query_len, heads, head_dim)`
        - ex. (128, 28, 8, 64)

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `torch.einsum()`
    - step. 8에서 확인

</details>

---

## [step. 12 head 결합 후 fc_out 적용]
head별 출력 (N, L, heads, head_dim)을 원래 임베딩 차원 (N, L, E)로 다시 합치고 마지막 선형 변환(fc_out)까지 적용함

### 1. 사용 함수
- `tensor.reshape()`, `nn.Linear()`

### 2. 패턴 예시
- `out = out.reshape(N, L, heads * head_dim)`
- `out = fc_out(out)`

### 3. 설명
    step. 11의 결과는 head별로 출력해서 마지막 두 축이 `(heads, head_dim)`으로 분리되어 있다.
    이걸 다시 하나로 합쳐 embed_size E를 복원해야 다음 레어어들이 `(N, L, E)`를 유지 할 수 있다.
    그래서 `(heads, head_dim)`을 reshape해 `(N, L, E)`로 만든다.
    그 다음 `fc_out(Linear(E, E))`을 한 번 더 적용해 head별로 계산된 정보를 다시 섞어 최종 self attention 출력으로 만든다
    
### 4. shape 변화
- 입력
    - head별 out : `(N, query_len, heads, head_dim)`
        - ex. (128, 28, 8, 64)
- 출력
    - reshape 후: `(N, query_len, heads * head_dim)` = `(N, query_len, E)`
        - ex. (128, 28, 512)
    - fc_out 후: `(N, query_len, E)`
        - ex. (128, 28, 512)

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `tensor.reshape()`
    - step. 7에서 확인

- `nn.Linear()`
    - step. 6에서 확인

</details>

---

## [step. 13 TransformerBlock 1단계]
Self attention 결과를 원본 입력(query)과 더해서(residual) 정보 손실을 막고 LayerNorm으로 안정화한 뒤 Dropout을 적용해서 다음 FFN으로 넘길 입력 x를 만듦

### 1. 사용 함수
- `nn.LayerNorm()`, `nn.Dropout()`

### 2. 패턴 예시
- `x = dropout(norm(attention + query))`

### 3. 설명
    Self attention 출력만 쓰면 입력(query)의 원래 정보가 약해지거나 학습이 불안정해질 수 있다.
    따라서 attention 출력에 입력을 그대로 더하는 residual 연결로 정보 흐름을 보존한다.
    그 다음 LayerNorm으로 각 토큰의 마지막 차원 E를 기준으로 값을 정규화해 학습을 안정화한다.
    마지막으로 Dropout을 적용해 과적합을 줄이고 다음 FFN으로 넘길 x를 만든다.
    
### 4. shape 변화
- 입력
    - query : `(N, L, E)`
        - ex. (128, 28, 512)
    - attention : `(N, L, E)`
        - ex. (128, 28, 512)
- 출력
    - x : `(N, L, E)`
        - ex. (128, 28, 512)

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `nn.LayerNorm()`
    - 입력 텐서를 특정 차원 기준으로 정규화(normalization) 하는 레이어
    - Pytorch 기본 함수가 아니다.

- `nn.Dropout()`
    - 학습 중 일부 뉴런 값을 확률적으로 0으로 만들어 overfitting을 방지하는 레이어
    - Pytorch 기본 함수가 아니다.

</details>

---

## [step. 14 TransformerBlock 2단계]
각 토큰 위치마다 독립적으로 적용되는 FFN(Feed Forward Network)을 통과시킨 뒤 Residual 연결과 LayerNorm으로 안정화하고 Dropout을 적용해서 TransformerBlock의 최종 출력 out을 만듦

### 1. 사용 함수
- `nn.Sequential()`, `nn.Linear()`, `nn.ReLU()`, `nn.LayerNorm()`, `nn.Dropout()`

### 2. 패턴 예시
- `forward = feed_forward(x)`
- `out = dropout(norm(forward + x))`

### 3. 설명
    FFN은 각 토큰 위치를 독립적으로 더 비선형적으로 변환해 표현력을 높이는 서브레이어이다.
    FFN 출력만 쓰면 입력 x의 정보가 약해질 수 있으므로 residual로 forward와 x를 더해 정보 흐름을 보존한다.
    그 다음 LayerNorm으로 값을 정규화해 학습을 안정화하고 Dropout으로 과적합을 줄여 TransformerBlock의 최종 out을 만든다.
    이 out이 Encoder와 Decoder에서 다음 블록으로 전달된다.
    
### 4. shape 변화
- 입력
    - x : `(N, L, E)`
        - ex. (128, 28, 512)
- 출력
    - out : `(N, L, E)`
        - ex. (128, 28, 512)

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `nn.Sequential(layer1, layer2, ... )`
    - 여러 레이어를 순서대로 연결해 하나의 모듈처럼 만드는 컨테이너
    - Pytorch 기본 함수가 아니다.
    - 입력 → layer1 → layer2 → layer3 → 출력 순서로 작동한다.

- `nn.Linear()`
    - step. 6에서 확인

- `nn.ReLU()`
    - ReLU 활성화 함수 레이어
    - `ReLU(x) = max(0, x)`로 입력값이 0보다 크면 그 값을 그대로 출력하고, 0 이하면 0을 출력한다.
    - Pytorch 기본 함수가 아니다.
    - 자세한 설명 : https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/
    - 예시
        ```
        relu = nn.ReLU()

        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = relu(x)

        print(y)
        ```

        결과 : `tensor([0., 0., 0., 1., 2.])`

- `nn.LayerNorm()`
    - step. 13에서 확인

- `nn.Dropout()`
    - step. 13에서 확인

</details>

---

## [step. 15 Encoder 레이어 반복 적용]
Encoder는 동일한 TransformerBlock을 num_layers 만큼 반복 적용해서 최종 encoder 출력 enc_out을 만듦

### 1. 사용 함수
- `nn.ModuleList()`

### 2. 패턴 예시
- `self.layers = nn.ModuleList([...])`
- `for layer in self.layers: out = layer(out, out, out, src_mask)`

### 3. 설명
    Encoder는 동일 구조의 TransformerBlock을 num_layers개 쌓아 순차적으로 적용한다.
    각 레이어는 입력 out을 더 풍부한 표현으로 갱신하고, 이 갱신이 반복되면서 문장 전체 문맥을 점점 더 잘 담는 enc_out을 만든다. 
    Encoder에서는 self attention이라 value key query가 모두 같은 out을 사용한다.
    최종 enc_out은 Decoder의 cross attention에서 key와 value로 제공된다.
    
### 4. shape 변화
- 입력
    - src 토큰 x : `(N, src_len)`
        - ex. (128, 28)
- 출력
    - enc_out : `(N, src_len, E)`
        - ex. (128, 28, 512)

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `nn.ModuleList([module1, module2, ...])`
    - 여러 개의 `nn.Module`을 리스트처럼 저장하는 컨테이너
    - 저장만 하는 컨테이너로 자동으로 forward를 실행해주지 않는다.
    - Pytorch 기본 함수는 아니다.

</details>

---

## [step. 16 DecoderBlock 1단계]
Decoder 입력 x(타깃 토큰 임베딩)에 대해 미래 토큰을 못 보게(trg_mask) 막은 masked self-attention을 수행하고 Residual + LayerNorm + Dropout으로 안정화한 query를 만듦

### 1. 사용 함수
- `SelfAttention()`, `nn.LayerNorm()`, `nn.Dropout()`

### 2. 패턴 예시
- `attention = self.attention(x, x, x, trg_mask)`
- `query = dropout(norm(attention + x))`

### 3. 설명
    Decoder는 다음 토큰을 생성하는 구조라 현재 위치에서 미래 토큰 정보를 보면 안된다.
    그래서 trg_mask를 넣은 masked self attention으로 x가 미래 위치를 참고하지 못하게 막는다.
    출력 attention에 입력 x를 residual로 더해 정보 흐름을 보존하고 LayerNorm으로 안정화한 뒤 Dropout을 적용해 query를 만든다.
    query가 다음 단계의 cross attention에서 Encoder 출력(enc_out)을 조회할 때 query 역할을 한다.
    
### 4. shape 변화
- 입력
    - x : `(N, trg_len, E)`
        - ex. (128, 29, 512)
    - trg_mask : `(N, 1, trg_len, trg_len)`
        - ex. (128, 1, 29, 29)
- 출력
    - query : `(N, trg_len, E)`
        - ex. (128, 29, 512)

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `selfAttention()`
    - Pytorch 기본 함수가 아닌, 우리가 구현해야 할 Self Attention 모듈 이름이다.
    - Self Attention은 입력 시퀀스 안에서 각 토큰이 같은 시퀀스의 다른 토큰들을 참조하는 Attention이다.

- `nn.LayerNorm()`
    - step. 13에서 확인
    
- `nn.Dropout()`
    - step. 13에서 확인

</details>

---

## [step. 17 DecoderBlock 2단계]
Decoder의 query가 Encoder 출력(enc_out)을 참고하도록 cross-attention을 수행하고, 그 뒤 FFN + Add&Norm까지 포함된 TransformerBlock을 통과시켜 DecoderBlock의 최종 출력 out을 만듦

### 1. 사용 함수
- `TransformerBlock()`

### 2. 패턴 예시
- `out = transformer_block(enc_out, enc_out, query, src_mask)`

### 3. 설명
    TransformerBlock 내부에서 SelfAttention(value, key, query, src_mask)로 cross attention을 계산하고, 
    FFN과 residual, LayerNorm, Dropout까지 적용해 최종 out을 만든다.
    src_mask는 Encoder 쪽 padding 위치를 cross attention에서 보지 못하게 막기 위해 사용한다.
    
### 4. shape 변화
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

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `TransformerBlock()`
    - Pytorch 기본 함수가 아닌, 우리가 구현해야 할 Encoder/Decoder 블록 모듈이다.

</details>

---

## [step. 18 Decoder 스택 반복 후 vocab logit 생성]
DecoderBlock을 num_layers 만큼 반복 적용해서 각 위치의 hidden state (N, trg_len, E)를 만들고 마지막에 fc_out 으로 vocab 차원으로 바꿔 최종 로짓(out)을 만듦

### 1. 사용 함수
- `DecoderBlock()`, `nn.Linear()`

### 2. 패턴 예시
- `for layer in layers: x = layer(x, enc_out, enc_out, src_mask, trg_mask)`
- `logits = fc_out(x)`

### 3. 설명
    Decoder는 DecoderBlock을 num_layers개 쌓아 반복 적용하면서 타깃 시퀀스의 hidden state를 점점 정교하게 만든다.
    반복이 끝나면 각 위치의 hidden state는 아직 임베딩 차원 E에 있으므로, 다음 토큰 분포를 만들기 위해 vocab 차원으로 만들어야 한다.
    그래서 fc_out(Linear(E, trg_vocab_size))을 적용해 각 위치마다 vocabulary 크기만큼의 점수 벡터, 즉 logits를 만든다.
    
### 4. shape 변화
- 입력
    - trg 토큰 x : `(N, trg_len)`
        - ex. (128, 29)
- 출력
    - decoder hidden x : `(N, trg_len, E)`
        - ex. (128, 29, 512)
    - out logits : `(N, trg_len, trg_vocab_size)`
        - ex. (128, 29, 10000)

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `DecorderBlock()`
    - Pytorch 기본 함수가 아닌, 우리가 구현해야 할 Encoder/Decoder 블록 모듈이다.
    - Decoder의 한 층을 구성하는 블록이다. (Self-Attention + Cross-Attention + FFN)
    
- `nn.Linear()`
    - step. 6에서 확인

</details>
    

---

## [step. 19 Transformer 전체 실행 흐름 결합]
Transformer.forward에서 src_mask, trg_mask를 만든 뒤 Encoder 출력(enc_src)을 Decoder에 전달해서 최종 logits(out)을 만듦

### 1. 사용 함수
- `make_src_mask()`, `make_trg_mask()`, `Encoder()`, `Decoder()`

### 2. 패턴 예시
- `src_mask = make_src_mask(src)`
- `trg_mask = make_trg_mask(trg)`
- `enc_src = encoder(src, src_mask)`
- `out = decoder(trg, enc_src, src_mask, trg_mask)`

### 3. 설명
    Transformer.forward로 전체 파이프라인을 한 번에 연결하는 단계이다.
    먼저 src padding mask와 trg causal mask를 만들어 어텐션이 보면 안 되는 위치를 차단한다.
    그 다음 Encoder가 src를 처리해 enc_src를 만들고, Decoder가 trg와 enc_src를 함께 사용해 다음 토큰 분포를 위한 logits(out)을 만든다.
    
### 4. shape 변화
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

### 5. 함수 설명
<details>
<summary> 함수 설명 </summary>

- `make_src_mask()`
    - Source 입력에서 padding 토큰을 가리는 mask를 만드는 함수
    - Pytorch 기본 함수가 아니다.
    - mask 생성 내용은 step. 3에 있다.

- `make_trg_mask()`
    - Decoder에서 padding 토큰을 가리고, 미래 토큰을 보지 못하게 하는 mask를 생성하는 함수
    - Pytorch 기본 함수가 아니다.
    - mask 생성 내용은 step. 3에 있다.

- `Encoder()`
    - Pytorch 기본 함수가 아닌, 우리가 구현해야 할 Transformer Encoder 모듈이다.

- `Decoder()`
    - Pytorch 기본 함수가 아닌, 우리가 구현해야 할 Transformer Decoder 모듈이다.

</details>
