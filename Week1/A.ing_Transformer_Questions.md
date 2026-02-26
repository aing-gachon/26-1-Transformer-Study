# Transformer 질문 정리 (Questions Only)
Attention is All you need 기반 질문 정리

---

## 1. 왜 Transformer는 단일 attention이 아니라 Multi-Head Attention(여러 head)을 사용해야 하는가?
논문 Section 3.2.2에서 Multi-Head Attention을 다음과 같이 정의한다.

$$MultiHead(Q, K, V) = Concat(head_1,\dots,head_h)W^O$$

각 head는 $Attention(QW_i^Q, KW_i^K, VW_i^V)$ 형태로 **서로 다른 선형변환(=서로 다른 관점)**을 가진다.

(1) 단일 Attention과 Multi-Head Attention의 차이를 설명하시오.  

(2) 단일 Attention이 아닌, Multi-Head Attention을 사용하는지 설명하시오.

---

## 2. Transformer는 Scaled Dot-Product Attention에서 $QK^\top$를 그대로 softmax에 넣지 않고 왜$\sqrt{d_k}$로 나눈 뒤 softmax를 적용하는가?
논문 Section 3.2.1은 $d_k$가 커질수록 dot-product 값이 커져 softmax가 **saturation(포화)** 될 수 있다고 설명한다.

(1) $QK^\top$의 분산 관점에서 scaling이 필요한 이유를 설명하시오.

(2) scaling을 하지 않을 경우 softmax의 saturation과 gradient에 어떤 문제가 발생하며, 이것이 학습 안정성과 어떤 관계가 있는지 설명하시오.

---

## 3. 왜 Transformer는 RNN과 Conv 대비 장기의존성(long-range dependency)에 유리하다고 주장하는가?
논문 Section 4와 Table 1은 self-attention, RNN(recurrent), CNN(convolution)의 **복잡도**와 **maximum path length**를 비교한다.

(1) Maximum Path Length의 의미를 설명하시오.

(2) Table 1의 Maximum Path Length 비교를 근거로, Self-attention이 long-range dependency 학습에 왜 유리한지 설명하시오.

---

## 4. Attention layer만 여러 층 쌓는 대신, FFN을 함께 사용하는 이유는 무엇인가?
Transformer는 각 layer에서 Multi-Head Attention 뒤에 position-wise Feed-Forward Network(FFN)을 둔다.  

$$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$$

(1) Transformer에서 Attention 뒤에 FFN을 두는 이유를 position-wise 비선형성 관점(각 위치마다 독립적으로 적용되는 비선형 변환)에서 설명하시오. 

(2) Attention sub-layer만 여러 층 쌓는 구조와 비교하여, FFN이 모델의 표현력을 어떻게 확장하는지 설명하시오.

---

## 5. Decoder의 self-attention에는 look-ahead mask가 필요한 이유는 무엇인가?
논문의 Section 3.2.3에서는 decoder의 self-attention에서 미래 위치를 참조하지 못하도록 masking을 적용한다고 명시한다.  

(1) auto-regressive 조건이 무엇인지 설명하시오.  

(2) look-ahead mask가 없으면 auto-regressive 조건이 어떻게 위배되는지 설명하시오.


---
