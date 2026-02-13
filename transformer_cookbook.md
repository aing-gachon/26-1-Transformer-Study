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