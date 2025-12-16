# 📘 Spotify Churn EDA Report (Synthetic Pattern-Injected Dataset)

> 본 EDA는 “현실 설명”보다 **의도한 이탈 패턴(가정)이 데이터에 제대로 주입되었는지 검증**하는 것을 목표로 한다.
> 원본 데이터(origin)는 변수–타깃 관계가 약해 학습이 어려웠고, 생성 데이터(generated)는 학습 가능한 신호를 만들기 위해 패턴을 주입했다.

---

## 1. Data Overview

### 1.1 데이터셋 규모

* Origin: **8,000 rows × 12 columns**
* Generated: **10,000 rows × 12 columns**
* Target: `is_churned` (1=이탈, 0=유지)

### 1.2 타깃 분포(이탈률)

* Origin 이탈률: **25.89%**
* Generated 이탈률: **31.79%**

📌 **해석 포인트**
Generated는 단순히 이탈률만 바뀐 게 아니라, **이탈이 특정 조건(광고/요금제/오프라인)에서 집중**되도록 구성되어 있음.

**[그림 자리]**

* `![타깃 분포(Origin vs Generated)](images/fig01_target_dist_compare.png)`

---

## 2. 데이터 생성(패턴 주입) 가정 요약

생성 코드에서 이탈 점수(`churn_score`)는 아래 “규칙”을 중심으로 구성됨:

1. **광고 피로(Free & ads > 15)** → 이탈 점수 **+4.0**
2. **콘텐츠 불만족(skip_rate > 0.6 & listening_time < 30)** → 이탈 점수 **+3.2**
3. **충성 유저(Premium or offline_listening=1)** → 이탈 점수 **-3.5**
4. 기본 성향:

   * `+ (skip_rate * 2.2)`
   * `- (songs_played_per_day * 0.05)`
5. 노이즈: `Normal(0, 1.5)` 추가

📌 **해석 포인트**
EDA에서는 “상관관계가 있다”로 끝내지 말고, **각 규칙이 실제로 데이터에서 관측되는지(검증)**로 서술하면 설득력이 커짐.

---

## 3. Data Quality Check (Generated)

### 3.1 결측치

* 결측치 비율: **전 컬럼 0% (없음)**

### 3.2 비현실 값(음수) 존재

정규분포로 생성된 변수에서 일부 음수 발생:

* `listening_time < 0` : **18건 (0.18%)**
* `songs_played_per_day < 0` : **228건 (2.28%)**

✅ **전처리 권장**

* 학습용 데이터로는 `clip(lower=0)` 또는 음수 행 제거 정책을 명시

**[그림 자리]**

* `![listening_time 분포 및 음수 확인](images/fig02_listening_time_hist.png)`
* `![songs_played_per_day 분포 및 음수 확인](images/fig03_songs_hist.png)`

---

## 4. 핵심 패턴 검증 결과(Generated vs Origin)

> 아래 3개는 Generated에서 **가장 강한 학습 신호**이며, Origin에서는 거의 차이가 없었다.

---

## 4.1 요금제(subscription_type)별 이탈률

### Origin (차이 약함)

* Free: **24.93%**
* Premium: **25.06%**
* Student: **26.19%**
* Family: **27.52%**

### Generated (차이 매우 큼)

* Free: **46.33%** (n=5,942 / 59.4%)
* Student: **26.57%** (n=1,020 / 10.2%)
* Premium: **5.10%** (n=3,038 / 30.4%)

📌 **해석 포인트**

* Generated에서는 “돈 낸 유저는 웬만하면 남는다”라는 가정이 **Premium 저이탈**로 뚜렷하게 구현됨.
* Origin에서 요금제별 이탈 차이가 거의 없었던 점과 대비되며, **학습 가능한 분리 신호가 만들어졌음을 확인**.

**[그림 자리]**

* `![요금제별 이탈률(Origin vs Generated)](images/fig04_subscription_churn_compare.png)`

---

## 4.2 오프라인 사용(offline_listening) 여부

### Origin

* offline=0: **24.93%**
* offline=1: **26.21%** (차이 미미)

### Generated

* offline=0: **45.59%** (n=5,012 / 50.1%)
* offline=1: **17.92%** (n=4,988 / 49.9%)

📌 **해석 포인트**

* 생성 규칙에서 offline=1은 “충성 유저”로 처리되어 이탈 점수 감소(-3.5) → **보호 요인(Protective factor)** 역할.
* EDA에서는 “오프라인 기능 사용 유도” 같은 액션으로 자연스럽게 연결 가능.

**[그림 자리]**

* `![오프라인 사용 여부별 이탈률(Origin vs Generated)](images/fig05_offline_churn_compare.png)`

---

## 4.3 광고 피로(ads_listened_per_week) 임계점(Threshold)

Generated에서 이탈률이 **ads=15를 기준으로 급격히 달라짐**:

* ads ≤ 15: **19.15%** (n=5,342 / 53.4%)
* ads ≥ 16: **46.29%** (n=4,658 / 46.6%)

Origin은 동일 기준에서 차이가 거의 없음:

* ads ≤ 15: **25.96%**
* ads ≥ 16: **25.60%**

📌 **해석 포인트**

* Generated의 광고 변수는 선형이 아니라 **임계점 이후 급증(Threshold jump)** 구조.
* 이 구조는 실제 운영에서도 “광고 빈도 제한/캡핑(capping)” 같은 정책 제안으로 연결하기 좋음.

**[그림 자리]**

* `![광고 노출 수 분포](images/fig06_ads_dist.png)`
* `![광고 노출 수별 이탈률(선/막대)](images/fig07_ads_churn_by_value.png)`
* `![ads 임계점(≤15 vs ≥16) 이탈률 비교(Origin vs Generated)](images/fig08_ads_threshold_compare.png)`

---

## 5. 상호작용(Interaction) — 세그먼트가 ‘진짜’ 갈리는 지점

### 5.1 요금제 × 오프라인 (Churn Heatmap 권장)

Generated churn rate:

* Free: offline=0 **66.33%**, offline=1 **26.26%**
* Student: offline=0 **46.67%**, offline=1 **6.47%**
* Premium: offline=0 **4.78%**, offline=1 **5.42%**

📌 **해석 포인트**

* Free/Student는 **offline 여부에 따라 이탈률이 크게 출렁임** → “기능 사용 유도”가 타겟팅 포인트.
* Premium은 이미 낮은 이탈률이라 offline 효과가 상대적으로 약함(천장효과 느낌).

**[그림 자리]**

![요금제×오프라인 이탈률 히트맵](images/fig09_sub_offline_heatmap.png)

---

## 6. 고위험/저위험 세그먼트 정의(실행 가능한 인사이트)

### 고위험 세그먼트 (Retention 최우선 타겟)

* **Free & ads ≥ 16 & offline=0**

  * 비중: **13.88% (1,388명)**
  * 이탈률: **92.07%**

### 저위험 세그먼트 (유지 강한 그룹)

* **Free & ads ≤ 15 & offline=1**

  * 비중: **15.99% (1,599명)**
  * 이탈률: **5.07%**

📌 **액션 제안 예시**

* Free 사용자 중 광고 과다 노출 그룹(≥16)에게:

  * 광고 캡핑 / 리워드형 광고로 전환 / Premium 체험 제공
* offline 미사용 그룹에게:

  * 오프라인 저장 기능 온보딩(튜토리얼/미션)로 “충성 행동” 유도

**[표/그림 자리]**

* `![세그먼트별 사용자수/이탈률 요약표](images/fig10_segment_summary_table.png)`

---

## 7. 결론(EDA Summary)

* Origin 데이터는 요금제/광고/오프라인 등 주요 변수에서 **이탈률 차이가 작아 학습 신호가 약함**.
* Generated 데이터는 데이터 생성 규칙에 따라

  1. **Free & 광고 과다(>15)**에서 이탈 급증(Threshold)
  2. **Premium 또는 offline 사용**은 이탈을 강하게 억제(Protective)
  3. 결과적으로 **세그먼트가 명확히 분리**되어 모델 학습에 유리한 구조를 가짐
* 단, 일부 연속형 변수에서 음수 값이 발생하므로 **전처리 정책을 명시**하는 것이 바람직함.

---

# 🧩 그림/표 배치 가이드 (추천 파일명 + 캡션 한 줄)

| Figure | 파일명(예시)                                | 캡션 한 줄(발표용)                                              |
| ------ | -------------------------------------- | -------------------------------------------------------- |
| Fig01  | `fig01_target_dist_compare.png`        | “Generated는 이탈률뿐 아니라 이탈이 특정 조건에 몰리도록 설계됨”                |
| Fig04  | `fig04_subscription_churn_compare.png` | “Origin은 요금제 차이가 거의 없지만 Generated는 Free↑, Premium↓로 분리됨” |
| Fig05  | `fig05_offline_churn_compare.png`      | “오프라인 사용은 강력한 이탈 억제 요인으로 작동”                             |
| Fig07  | `fig07_ads_churn_by_value.png`         | “광고는 선형이 아니라 임계점 이후 이탈이 점프하는 구조”                         |
| Fig09  | `fig09_sub_offline_heatmap.png`        | “요금제×오프라인 조합에서 고위험 세그먼트가 선명하게 드러남”                       |
| Fig10  | `fig10_segment_summary_table.png`      | “Free·광고과다·오프라인미사용 그룹의 이탈률이 90%대로 가장 위험”                 |

