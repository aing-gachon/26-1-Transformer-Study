# Attention 치트시트 초안

## Shape 표기 규칙

- $$N$$ (배치 크기(batch size)) - 한번에 계산되는 샘플(예시: 문장, 문서 등)의 크기  
- $$src_{len}$$ (소스 길이(source length)): 인코더에 들어가는 한 샘플의 토큰의 개수  
- $$trg_{len}$$ (타깃 길이(target length)): 디코더에 들어가는 한 샘플의 토큰의 개수  
- $$q_{len} / k_{len} / v_{len}$$ (query/key/value length): 각 $$Q,K,V$$가 가진 토큰의 수  
- $$d_{model}$$ ($$embed_{size}$$): 한 토큰이 임베딩을 거치고 표현되는 기본 차원의 크기  
- $$h$$ (heads): Multi-Head Attention에서 각각 독립적으로 attention을 계산하는 하나의 attention 채널의 개수  
- $$d_{k}$$ ($$head_{dim} = d_{model} / h$$): 각 head에서 $$Q/K/V$$가 가지는 차원  

---

## 0) 전체 구조

<img width="536" height="736" alt="image" src="https://github.com/user-attachments/assets/9ba5c565-77b4-4dd7-8055-e603e40447e2" />


---

## 1) Scaled Dot-Product Attention ↔ `SelfAttention.forward`
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/0553682f-89f5-41f8-94b7-1a5db0fa2659" />


| 요소 | 논문 수식/그림 | 코드 좌표 | 핵심 설명 |
| --- | --- | --- | --- |
| $$Q = XW^{Q}$$ | Eq.(1), 3.2.1~3.2.2 | `queries = self.queries(query)` | **Query**: 내가 지금 찾고 싶은 정보의 기준(질문)<br>$$X$$: 현재 레이어 입력(임베딩+포지션 또는 이전 레이어 출력)<br>$$W^{Q}$$: 질문에 필요한 특징만 뽑는 필터(선형변환)<br>왜 필요한가: 같은 $$X$$라도 비교용(질문) 관점으로 재표현해야 다음 단계에서 **Key**와 내적 비교가 의미 있어짐. |
| $$K = XW^{K}$$ | Eq.(1), 3.2.1~3.2.2 | `keys = self.keys(keys)` | **Key**: 내가 어떤 질문에 잘 맞는지 나타내는 주소<br>$$W^{K}$$: 각 토큰을 매칭용 특징으로 바꿈<br>왜 필요한가: Query가 찾는 조건이라면 Key는 “나는 이런 성질을 가짐”이라서 $$Q$$와 $$K$$를 같은 기준 공간에서 비교할 수 있게 함. |
| $$V = XW^{V}$$ | Eq.(1), 3.2.1~3.2.2 | `values = self.values(values)` | **Value**: 실제로 가져와 섞을 내용<br>$$W^{V}$$: 가져올 정보의 형태를 정리(가공)하는 역할<br>왜 필요한가?: attention은 결국 $$V$$의 가중합을 출력으로 내기 때문에 $$V$$가 “내가 제공할 내용”이 되어야 함. |
| head split | 3.2.2, Fig.2 | `reshape(N, len, heads, head_dim)` | 한 번에 한 관점만 보면 한계 → 여러 관점으로 병렬 탐색<br>$$d_{model}$$을 $$h$$개의 작은 공간 $$d_{k}$$로 나눔: $$[N, L, d_{model}] \rightarrow [N, L, h, d_{k}]$$<br>왜 필요한가: 어떤 head는 근거리, 어떤 head는 문법 관계, 어떤 head는 의미 관계처럼 서로 다른 패턴을 동시에 잡도록 학습될 수 있음. |
| $$QK^{T}$$ | Eq.(1) | `energy = einsum("nqhd,nkhd->nhqk", ...)` | 유사도 점수표(스코어 보드)<br>각 query 위치가 각 key 위치를 얼마나 봐야 하는지 점수화<br>직관: $$q$$번째 토큰이 $$k$$번째 토큰을 얼마나 참고할까?를 head별로 계산한 표 |
| mask | 3.2.3 | `masked_fill(mask == 0, -1e20)` | 보면 안 되는 칸을 시험지에서 가려버리는 장치<br>**padding mask**: 의미 없는 padding 토큰에 점수 주는 걸 방지(성능/안정성↑)<br>**causal mask**(디코더): 미래 토큰을 보면 정답을 미리 보는 치팅 → 학습이 무너짐 방지. softmax 전에 $$-\infty$$로 만들어 해당 위치 확률이 0이 되게 함 |
| $$\frac{QK^{T}}{\sqrt{d_{k}}}$$ | Eq.(1) | `softmax(energy / sqrt(dk), dim=3)` | softmax가 한쪽으로 과하게 쏠리는 것을 막는 스케일링<br>$$d_{k}$$가 크면 내적 값 분산이 커져 softmax가 거의 0/1로 포화 → gradient 약해짐<br>$$\sqrt{d_{k}}$$로 나눠 분포를 완만하게 만들어 학습이 안정됨 |
| softmax(...) | Eq.(1) | `attention = torch.softmax(..., dim=3)` | 점수표 → 확률표(가중치표)<br>각 query마다 key들에 대해 합이 1인 분포를 만듦<br>직관: 참고 비율(얼마나 볼지) |
| softmax(...) $$V$$ | Eq.(1) | `out = einsum("nhql,nlhd->nqhd", ...)` | 참고 비율로 ‘내용($$V$$)’을 섞어 새 표현을 만든다<br>attention이 크면 그 토큰의 $$V$$가 많이 섞임<br>결과: query 위치마다 필요한 정보만 모아온 요약 벡터 |
| $$(Concat(heads)W^{O})$$ | 3.2.2, Fig.2 | `reshape(..., h*d_k) → fc_out` | 여러 관점 결과를 합쳐 한 개의 표현으로 정리<br>concat: 서로 다른 head 정보 손실 없이 붙임 $$[N, q_{len}, h, d_{k}] \rightarrow [N, q_{len}, d_{model}]$$<br>$$W^{O}$$: 어떤 head 정보를 어떻게 섞을지 학습하는 최종 정리 단계 → 다음 블록(Add&Norm/FFN)이 기대하는 차원으로 맞춤 |

---

## 2) Multi-Head Attention (MHA) ↔ split → attention → concat → projection
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/a1f01a75-5647-4cbd-b5c4-aa747c40d052" />


| 요소 | 논문 수식/그림 | 코드 좌표 | 핵심 설명 |
| --- | --- | --- | --- |
| $$h$$ (heads) | 3.2.2 | `self.heads = heads` | attention을 여러 번 작게 병렬로 돌리는 개수<br>한 head만 쓰면 한 가지 관계에 과적합/한계<br>여러 head로 나누면 서로 다른 관계를 동시에 학습(예: 장거리/근거리, 주어-동사, 수식 관계 등) |
| $$d_{k} = \frac{d_{model}}{h}$$ | 3.2.2 | `head_dim = embed_size // heads` | 총 계산량을 크게 늘리지 않으면서 head 수를 늘리는 트릭<br>$$h$$개로 나누면 head당 차원이 줄어듦. 결과적으로 표현력(관점 수)↑ / 연산량 균형 유지 |
| $$Concat(head_1..head_h)$$ | 3.2.2, Fig.2 | `reshape(N, q_len, h*d_k)` | head별 결과를 정보 보존 형태로 합침<br>평균을 내면 head별 특성이 섞여 사라질 수 있음. concat은 각 head의 시각을 그대로 모아 다음 단계에서 선택적으로 조합 가능 |
| $$W^{O}$$ (output projection) | 3.2.2 | `fc_out` | concat된 정보를 다시 $$d_{model}$$ 공간으로 정돈<br>concat은 단순히 붙인 거라 잘 섞인 표현이 아님<br>$$W^{O}$$가 어떤 head 정보를 얼마나 섞을지 학습 → 레이어 출력으로 쓸 수 있는 형태로 변환 |

---

## 3) Encoder / Decoder의 3가지 Attention 매핑

| 요소 | 논문 수식/그림 | 코드 좌표 | 핵심 설명 |
| --- | --- | --- | --- |
| Encoder Self-Attention | 3.2.3, Fig.1/2 | `layer(out, out, out, src_mask)` | 입력 문장 내부에서 문맥 만들기<br>같은 문장 안 토큰들이 서로를 참고해 문맥화된 표현 생성<br>예: bank가 강/은행 중 무엇인지 주변 단어를 보고 결정<br>$$Q = K = V$$ = 인코더 상태 |
| Decoder Masked Self-Attention | 3.2.3, Fig.2 | `attention(x, x, x, trg_mask)` | 생성 모델의 규칙(미래 금지)을 지키는 self-attention<br>$$i$$번째 출력은 $$0..i-1$$까지만 봐야 함<br>마스크 없으면 학습 때 정답을 보게 돼서 실제 생성(inference) 때 성능이 무너짐 |
| Encoder-Decoder Attention (Cross-Attention) | 3.2.3, Fig.2 | `value=enc_out, key=enc_out, query=dec_state` | 지금 출력할 토큰을 위해 입력의 어디를 참고할까?<br>Query는 디코더(현재 생성 상태)<br>Key/Value는 인코더 출력(입력 문장의 정보 창고)<br>직관: 출력 단어를 만들 때 입력 문장 특정 단어/구를 가리키며 참고 |

---

## 4) Mask(마스크) ↔ `make_src_mask`, `make_trg_mask`

| 요소 | 논문 수식/그림 | 코드 좌표 | 핵심 설명 |
| --- | --- | --- | --- |
| 소스 패딩 마스크 (padding mask) | 3.2.3 | `make_src_mask` | 길이 맞추기용 padding은 정보가 0이므로 무시해야 함<br>안 막으면 모델이 padding에도 확률을 줘서 학습이 흔들림. 특히 긴 padding이 많으면 attention이 의미 없는 곳으로 새는 현상 발생 |
| 타깃 이후 마스크 (subsequent/causal mask) | 3.2.3 | `make_trg_mask` | 미래 토큰 정보 유출 방지(치팅 방지)<br>학습: 정답 문장을 다 알고 있으니 마스크 없으면 미래를 봐버림<br>추론: 미래 토큰은 존재하지 않음 → 학습/추론 조건 불일치 문제를 막는 핵심 장치 |

---

## 5) Add & Norm + FFN (Eq.2) ↔ `TransformerBlock.forward`

| 요소 | 논문 수식/그림 | 코드 좌표 | 핵심 설명 |
| --- | --- | --- | --- |
| Add & Norm (after attention) | Fig.2, 3.1 | `norm1(attn + query)` | Residual + LayerNorm의 조합 = 학습이 잘 되게 하는 기본 골격<br>Residual(+): attention이 실패해도 원래 정보(query)가 살아남아 학습이 안정<br>LayerNorm: 값 분포를 정리해서 다음 연산이 안정<br>Dropout: 과적합 방지(특정 경로에만 의존하는 걸 막음) |
| $$FFN(x)=\max(0, xW_1+b_1)W_2+b_2$$ | Eq.(2), 3.3 | `Linear → ReLU → Linear` | 토큰별로 적용되는 작은 MLP(표현력 증폭기)<br>attention이 “토큰 간 정보 섞기”라면 FFN은 섞인 결과를 비선형 변환해 더 풍부한 특징으로 바꿈<br>position-wise라서 각 위치는 독립 처리(하지만 attention으로 이미 문맥이 들어있음) |
| Add & Norm (after FFN) | Fig.2, 3.1 | `norm2(ffn + x)` | FFN도 깊게 쌓이므로 동일한 안정화 장치 적용<br>FFN이 만든 변화량만 추가하고(Residual) 분포를 다시 정리(LN)해서 다음 레이어로 넘김 |

---

## 6) Embedding + Positional Encoding ↔ `Encoder/Decoder.forward`

| 요소 | 논문 수식/그림 | 코드 좌표 | 핵심 설명 |
| --- | --- | --- | --- |
| token embedding | 3.4 | `word_embedding(x)` | 정수 ID → 의미 벡터로 변환<br>모델은 숫자 ID 자체의 크기에 의미를 두면 안 됨(“3이 7보다 작다” 같은 의미는 없음)<br>embedding은 “단어/토큰의 의미”를 학습해 벡터 공간에 배치 |
| positional encoding (learned) | 3.5 | `position_embedding(positions)` | 순서 정보 주입(Transformer의 필수 보완재)<br>RNN/Conv는 구조적으로 순서가 반영되지만, self-attention은 구조만 보면 순서가 없음<br>그래서 몇 번째 토큰인지를 벡터로 만들어 의미 벡터에 합침 |
| sum + dropout | 3.4~3.5 | `dropout(word + pos)` | 의미(무슨 단어) + 위치(몇 번째)를 한 벡터로 통합<br>합(sum)은 차원을 늘리지 않고 자연스럽게 결합하는 가장 흔한 방법<br>dropout은 학습 시 특정 단서에만 의존하는 걸 줄여 일반화 도움 |

---

## 7) Output projection (Logits) ↔ `Decoder.fc_out`

| 요소 | 논문 수식/그림 | 코드 좌표 | 핵심 설명 |
| --- | --- | --- | --- |
| vocab projection → logits | 3.4 | `fc_out(d_model → vocab)` | 다음 토큰 후보들에 점수 매기기<br>디코더 출력은 의미 공간 벡터($$d_{model}$$)라서 바로 단어를 고를 수 없음. vocab 크기만큼 점수(logits)를 만들고, softmax로 확률로 바꿔 예측<br>학습은 정답 토큰의 확률을 높이도록(=loss 최소화) 진행 |
