# Transformer 질문 정리 (Questions and Answer) 
Attention is All you need 기반 질문 답변


## 1. 왜 Transformer는 단일 attention이 아니라 Multi-Head Attention(여러 head)을 사용해야 하는가?
논문 Section 3.2.2에서 Multi-Head Attention을 다음과 같이 정의한다.

$$MultiHead(Q, K, V) = Concat(head_1,\dots,head_h)W^O$$

각 head는 $Attention(QW_i^Q, KW_i^K, VW_i^V)$ 형태로 **서로 다른 선형변환(=서로 다른 관점)** 을 가진다.

### (1) 단일 Attention과 Multi-Head Attention의 차이를 설명하시오.
#### ① 단일 Attention (Single-Head Attention)
- **한 번만** attention을 수행한다.
- 하나의 $Q,K,V$ 투영(projection)으로 만들어진 **단 하나의 attention 분포(가중치 행렬)** 로 value들을 가중합한다.
- 결과적으로 어디를 볼지(가중치)도 어떤 특징을 볼지(표현공간)도 **한 세트**로만 결정된다.

- 수식으로 보면
  
  $$Attention(Q,K,V)=softmax\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
  
  이고, 여기서 softmax 뒤에 곱해지는 $V$의 **가중합이 최종 출력**이다.  
- 즉, 단일 head는 **한 장의 정렬표(alignment map) + 그 정렬표로 만든 한 번의 평균(averaging)** 이라고 볼 수 있다.

#### ② Multi-Head Attention
- $Q,K,V$를 **h개 head마다 서로 다른 선형변환**($W_i^Q, W_i^K, W_i^V$)으로 서로 다른 표현 부분공간(subspace)[[1]](#note-1)을 투영한다.
- 각 head는 **독립적으로** scaled dot-product attention을 수행해 **h개의 서로 다른 attention 분포**를 만든다.
- 각 head 출력들을 **concat**하여 붙이고, 마지막에 $W^O$로 다시 섞어 최종 출력으로 만든다.

- 차원의 관점(논문 설정)으로 보면, 보통 $d_k=d_v=d_{model}/h$이다.  
  각 head는 더 작은 차원에서 일하지만, h개를 병렬로 하므로 **총 연산량은 대체로 비슷**하게 유지된다.
    - Ex) $d_{model}=512, h=8 \Rightarrow d_k=d_v=64$.

<a id="note-1"></a>
**[1] 서로 다른 표현 부분공간(subspace)이란?**  
- 입력 벡터 $x\in\mathbb{R}^{d_{model}}$를 head마다 다른 행렬 $W_i$로 곱해 $\mathbb{R}^{d_k}$로 보내는 것.
- 즉 **같은 토큰이라도 head마다 ‘특징을 보는 렌즈’가 다르다**는 뜻이다.

---

### (2) 왜 단일 Attention이 아니라 Multi-Head Attention을 사용하는지 설명하시오.
- Attention의 출력은 결국 $V$들의 **가중합**이다.  
- 단일 head는, **여러 종류의 관계(예: 구문적 관계, 의미적 대응, 거리/위치 패턴)** 를 한 장의 가중치 행렬에 억지로 담아야 한다.
- 이때 가장 큰 문제는 논문 표현 그대로 **averaging(평균)이 모든 관계 정보를 섞어버린다**는 점이다.  
  - 예를 들어 어떤 단어는 주어-동사 관계를 강하게 보면서 동시에 수식어-피수식어 관계도 강하게 봐야 할 수 있는데,  
    단일 head는 한 번의 평균으로 둘을 함께 만족시키기 어렵다(타협이 필요).
- 이처럼 단일 head는 가중합 평균이 **한 번**이라서 서로 다른 관계를 동시에 담기 어려운 반면,
  Multi-Head는 **여러 관계를 병렬로 분리해서** 학습[[2]](#note-2)하기 때문에 적합하다.  
- 결과적으로 Multi-Head를 통해 여러 종류의 의존성/특징을 병렬로 포착할 수 있고, 단일 head에서 생기는 평균으로 인한 정보 손실을 줄인다.  
- 또한 논문에서 실험(Section 6.2, Table 3 row (A))한 결과, head 수를 8→1로 줄이면 성능(BLEU)이 하락(약 0.9 BLEU)한다.  

<a id="note-2"></a>
**[2] Multi-Head가 해결하는 방식**  
- head A는 문법적 연결(예: 주어→동사)에 집중    
- head B는 의미적 연결(예: 대명사→선행사)에 집중  
- head C는 가까운 토큰/먼 토큰 등 거리 패턴에 집중  
- 과 같이 **역할 분담**하여 관계 정보를 학습한다.

**논문 근거**  
- Section 3.2.2 -> "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."
- Section 6 -> "While single-head attention is 0.9 BLEU worse than the best setting."

---

## 2. Transformer는 Scaled Dot-Product Attention에서 $QK^\top$를 그대로 softmax에 넣지 않고 왜 $\sqrt{d_k}$로 나눈 뒤 softmax를 적용하는가?
논문 Section 3.2.1은 $d_k$가 커질수록 dot-product 값이 커져 softmax가 **saturation(포화)** 될 수 있다고 설명한다.

### (1) $QK^\top$의 분산 관점에서 scaling이 필요한 이유를 설명하시오.
- 내적 $q\cdot k$는 차원 $d_k$가 커질수록 분산이 $d_k$에 비례해 커진다[[3]](#note-3).
- softmax에 들어가는 값이 커지면, 지수함수 특성상 출력이 한쪽으로 치우치는 문제가 생긴다.
- 따라서 $1/\sqrt{d_k}$로 scaling하여 크기를 정상화한다.

<a id="note-3"></a>
**[3] 왜 분산이 $d_k$에 비례하나?**  
- $q_i, k_i$가 평균 0, 분산 1인 독립 변수라고 가정하면,
  
$$q\cdot k = \sum_{i=1}^{d_k} q_i k_i$$
  
- 합의 분산은 더해지므로 $Var(q\cdot k)=d_k$.  
  즉, **표준편차(전형적인 크기)** 는 $\sqrt{d_k}$ 정도로 커진다.
- 그래서 $\frac{q\cdot k}{\sqrt{d_k}}$로 만들면 분산이 $O(1)$ 수준으로 유지되어 softmax에 들어가는 값의 스케일이 안정적이다.

**논문 근거**  
- Section 3.2.1 -> "We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $1/√d_k$."

---

### (2) scaling을 하지 않을 경우 softmax saturation과 gradient에 어떤 문제가 발생하며, 이것이 학습 안정성과 어떤 관계가 있는지 설명하시오.
#### softmax saturation 관점
- softmax saturation은 입력 $z$의 값 차이가 너무 커서, 출력 확률 $p$가 거의 **one-hot(한 곳만 1에 가까움)이 되는 현상**이다.
  - 예를 들어, 한 logit이 10, 나머지가 0이라면 $e^{10}$이 너무 커서 거의 한 위치에 확률이 몰린다.
- scaling을 하지 않으면, 출력이 $d_k$과 비례하게 커지고 극단적으로 치우쳐서 softmax saturation 문제가 생긴다.  

#### gradient 관점
- softmax의 미분은 $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij}-p_j)$이다. 
- saturation이면 어떤 $p_i\approx 1$, 나머지는 $\approx 0$이어서  
  - $p_i(1-p_i)\approx 0$  
  - $p_i p_j\approx 0$  
  가 되어 대부분의 항이 **거의 0**이 된다.
- 이로 인해 gradient가 매우 작아지고 역전파 신호가 약해진다.(학습 느려짐/불안정)

#### 학습 안정성 관점
- 초기에 attention이 너무 날카로워지면(거의 하나만 선택) 학습이 탐색을 잘 못하고 특정 패턴에 과하게 고정될 수 있다.
- Scaling은 attention 분포를 적절히 부드럽게 만들고, gradient가 너무 작아지는 문제를 줄여 **학습을 안정화**한다.

**논문 근거**
- Section 3.2.1 -> "We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients.

---

## 3. 왜 Transformer는 RNN과 Conv 대비 장기의존성(long-range dependency)에 유리하다고 주장하는가?
논문 Section 4와 Table 1은 self-attention, RNN(recurrent), CNN(convolution)의 **복잡도**와 **maximum path length**를 비교한다.

### (1) Maximum Path Length의 의미를 설명하시오.
- Maximum Path Length는 입력 시퀀스에서 임의의 두 위치(토큰) 사이의 정보가 전달되기 위해 거쳐야 하는 **최대 연산 단계 수(최장 경로 길이)** 를 뜻한다.
  - 쉽게 말해, 토큰 A의 정보가 토큰 B에 영향을 주려면 네트워크 안에서 몇 번의 층/시간단계를 거쳐야 하는가?와 같다.
- 이 경로가 길수록 forward 신호도, backward gradient도 더 많은 변환을 거치므로 **long-range 관계 학습이 어려워질 수 있다.**

**논문 근거**
- Section 4 -> "We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients.

---

### (2) Table 1의 Maximum Path Length 비교를 근거로, Self-attention이 long-range dependency 학습에 왜 유리한지 설명하시오.
- **Self-attention:** $O(1)$  
  한 층에서 모든 위치가 서로를 직접 볼 수 있어, 토큰 i → 토큰 j가 **바로 연결**된다.
- **RNN:** $O(n)$  
  i → j로 가려면 중간의 모든 time step을 순서대로 거쳐야 한다(**순차 처리**).
- **CNN:** 보통 $O(\log_k n)$ (dilated 등 설계에 따라)  
  한 층의 receptive field가 제한적이라 멀리 보려면 **여러 층을 쌓아야 한다**.
- 따라서 self-attention은 장거리 관계를 멀리 돌아서 전달하지 않고 짧은 경로로 바로 연결해 장기의존성을 더 쉽게 학습[[4]](#note-4).할 수 있다.

<a id="note-4"></a>
**[4] 왜 경로가 짧으면 유리한가?**  
- **정보 병목(bottleneck) 감소:**  
  - RNN은 hidden state 하나에 정보를 계속 누적해야 해서 먼 관계가 압축되기 쉽다.  
  - self-attention은 필요한 토큰을 직접 참조하므로 압축 부담이 줄어든다.
- **gradient 전달이 쉬움:**  
  - 경로가 길면(많은 단계) gradient가 희석/왜곡될 기회가 많다.
  - 경로가 짧으면(한두 단계) 필요한 신호가 더 직접적으로 전달된다.
- **전역 문맥 접근:**  
  - 문장 내 멀리 떨어진 주어-동사 관계, 대명사-선행사 관계 등을 한 층에서 바로 연결 가능하다.

**논문 근거**
- Section 4 -> “The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies.” 

---

## 4. Attention layer만 여러 층 쌓는 대신, FFN을 함께 사용하는 이유는 무엇인가?
Transformer는 각 layer에서 Multi-Head Attention 뒤에 position-wise Feed-Forward Network(FFN)을 둔다.  

$$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$$

### (1) Transformer에서 Attention 뒤에 FFN을 두는 이유를 position-wise 비선형성 관점(각 위치마다 독립적으로 적용되는 비선형 변환)에서 설명하시오.
- Attention 출력은 (가중치가 고정되면) 기본적으로 $V$의 **가중합(선형 결합)** 이다.
- FFN은 각 토큰 위치마다 동일한 MLP(선형→ReLU→선형)를 적용해 **차원 간 상호작용 + 비선형 변환**을 추가한다.
- 논문은 FFN을 각 position에 동일하게 적용되는 fully connected network라고 설명하며, 두 선형변환 사이에 ReLU가 있다고 말한다(Section 3.3).
- 결론적으로, Attention이 정보를 **어디서 가져올지(섞을지)** 를 정하고, FFN은 각 위치에서 가져온 정보를 **어떻게 변환할지(비선형 처리)** 를 담당한다.

**직관적인 비유**  
- Attention: 필요한 정보를 메모리에서 찾아 붙이는 라우팅/검색
- FFN: 붙여진 정보를 가지고 그 위치에서 계산/해석하는 작은 뇌(MLP)

**논문 근거**
- Section 3.3 -> "In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between."

---

### (2) Attention sub-layer만 여러 층 쌓는 구조와 비교하여, FFN이 모델의 표현력을 어떻게 확장하는지 설명하시오.
- Attention을 통해 여러 층을 쌓으면 서로 다른 위치 정보는 계속 섞이지만, 
  **각 토큰 벡터 내부($d_{model}$ 차원)에서의 복잡한 변환**은 상대적으로 약해질 수 있다.
- FFN을 사용하면 각 위치 표현을 고차원 공간에서 비선형적으로 변환할 수 있어 모델의 표현력이 확장된다[[5]](#note-5).  

<a id="note-5"></a>
**[5] FNN의 장점**  
- **비선형성 추가:** ReLU가 들어가면서 표현력이 크게 증가한다.
- **확장-축소 구조:** $d_{model}\to d_{ff}\to d_{model}$ (논문 base: 512→2048→512)  
  잠깐 큰 공간으로 보내서 더 풍부한 특징을 만들고 다시 요약한다.
- **역할 분리:** Attention은 토큰 간 상호작용(문맥 결합), FFN은 토큰별 특징 추출/변환(내용 가공)으로 분업되어 학습이 더 잘 된다.

**논문 근거**  
- Section 3.2 -> "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key."

---

## 5. Decoder의 self-attention에는 look-ahead mask가 필요한 이유는 무엇인가?
논문 Section 3.1과 3.2.3에서 decoder self-attention이 **미래 토큰을 보지 못하도록(masking)** 해야 한다고 말한다.

### (1) auto-regressive 조건이 무엇인지 설명하시오.
- Auto-regressive란 출력 시퀀스 확률을 다음처럼 인수분해하는 생성 방식이다.
    
$$p(y|x)=\prod_{t=1}^{m} p(y_t\mid y_{<t}, x)$$

- 즉, **t번째 토큰 예측은 이전 토큰들($y_{<t}$)과 입력($x$)에만 의존**해야 한다.

**논문 근거**
- Section 3 -> "At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next."

---

### (2) look-ahead mask가 없으면 auto-regressive 조건이 어떻게 위배되는지 설명하시오.
- mask가 없으면, decoder self-attention에서 위치 $t$가 $t+1, t+2,\dots$ 같은 **미래 정답 토큰**을 볼 수 있다.  
- 그러면 $p(y_t\mid y_{<t},x)$가 아니라 $p(y_t\mid y_{\le m}, x)$처럼 정답을 미리 본 조건부 확률을 학습하게 된다. (쉽게 말하면 cheating)

- **훈련 시(teacher forcing)** 에는 정답 시퀀스 전체가 입력으로 들어오므로, 미래를 볼 수 있으면 모델이 치팅한다.
- **추론(inference)** 때는 미래 토큰이 존재하지 않으므로, 훈련 때 배운 방식과 상황이 달라져 성능이 크게 떨어질 수 있다(훈련-추론 불일치).

#### 구현 관점(어떻게 막나?)
- attention logits $QK^\top/\sqrt{d_k}$에서 미래 위치에 해당하는 값을 $-\infty$로 만든 뒤 softmax를 적용한다.
- softmax 결과에서 그 위치의 확률이 0이 되어 미래를 참조할 수 없다.
- 논문은 masking + output embeddings를 한 칸 offset을 결합해 position i가 오직 i보다 작은 위치만 보도록 한다(Section 3.1).

**논문 근거**  
- Section 3 -> "At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next."
- Section 3.1 -> "We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i."

---


