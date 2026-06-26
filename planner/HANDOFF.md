# 핸드오프 — 이종 TP/PP 서빙 플래너

> ⚠️ **이 파일은 세션별 작업 로그입니다 (날짜별 체크포인트 수치 포함, 일부는 옛 fit 상태).**
> **현재 권위 있는 상태/수치는 `README.md` + `planner_describe.md`(특히 §6.3, §8) 를 보세요.**
> 최신 요약(2026-06-26): plan_safe 제거→raw argmax; qwen3-32B는 fork PP-overlap 서빙 갭으로
> 평가서 제외(근본 규명, §8); 4-모델 헤드라인 +31.4% mean / 38·41; `--validate` regret 3.3%
> (median 0%, n=1 probe가 평균 끌어올림); check_consistency 17개.

## 큰 그림
4+4 cross-node 클러스터(head=4×Blackwell 96GB, worker=4×Ada 48GB, IB)에서
non-uniform TP/PP 서빙을 측정하고, 그 데이터로 **임의의 모델·하드웨어·워크로드에서
최적 (tp, pp, layer_split, ffn/head split) 을 예측하는 해석적 플래너**를 만드는 게 목표.
논문 메시지: "hybrid parallelism is not always best" — 워크로드/모델마다 최적 토폴로지가 다르다.

## 완료된 것 ✅
- **4 모델 full sweep** (각 15~19 config × 3 workload): Llama-3.1-8B, Llama-3.3-70B,
  OPT-30B(MHA, n_req=64), Qwen3-32B. 결과: `results/hetero_4x4_<model>_full_<ts>/`
- **PP overlap 검증** (nsight + torch profiler): multi-stream concurrency, cross-stage
  NCCL P2P 9137 kernels, 양 노드 동시 busy, 실제 응답 정상. M13 broadcast_stream +
  microbatch(auto-tuner) 실작동 확인. 검증 코드: `planner/verify_pp_overlap_torch_profiler.py`,
  `planner/analyze_pp_overlap_trace.py`
- **calibration_data.csv** (211 rows): model, config, splits(B:A per-rank), workload,
  tps/ttft/itl, regime(stock/overlap/tp_only), 모델 spec 컬럼. 재생성: `planner/build_calibration.py`
- **hw_params.json**: 측정 역산 effective 하드웨어 파라미터 (Ada decode BW 707 GB/s,
  Blackwell 1400; prefill TFLOPS Ada 183 / BW 289; cross-node AR ~350µs; overlap gain 등)
- **PLANNER_SPEC.md**: roofline + 비균등 TP 닫힌형 최적분할 + PP steady-state + 메모리
  feasibility + 신규 클러스터 캘리브레이션(≤6 probe) 수식 일체
- **perf_planner.py**: predict_tps / plan / validate / CLI / predict_mistral(사전등록)
- **fit_planner.py**: 7 자유 파라미터 robust fit + leave-one-model-out
- **figures/**: model별 4 topology plot ×4, 4-model 통합, 70B stock-vs-overlap

## 측정 핵심 결과 (overall champion, TP8PP1 uniform 대비)
| Model | balanced | decode | prefill |
|---|---|---|---|
| 8B | PP2 TP4 +89% | PP2 TP4 +125% | PP2 TP4 +40% |
| 70B | PP2 TP4 +24% | PP2 TP4 +15% | **TP8 FFN bias +18%** |
| OPT-30B | PP2 TP4 +94% | PP2 TP4 +155% | PP2 TP4 +133% |
| Qwen3-32B | PP2 TP4 +152% | PP2 TP4 +147% | PP2 TP4 +139% |
→ 거의 항상 PP2(=TP4PP2 overlap)가 이기지만 70B prefill만 TP8. 비균등은 +5~25% 보너스.

## 2026-06-13 세션 업데이트 (planner 정밀화 + Mistral 사전등록 완료)
**플래너 정확도 — 진단 후 수정 완료. 평가 지표를 regret 중심으로 재정의.**
- champion top-1 match는 **noise-limited** (PP2 skew 곡선이 top 근처 <5%, 측정노이즈와
  동급; opt30b decode→uniform vs 70b decode→skew+16 처럼 모델 간 추세까지 상충).
  → 논문 관례대로 **regret(optimality-gap) 1순위, Spearman/top-3 보조, MAPE는 예측기 정확도**.
- **fit v8 결과**: mean regret **8.5%** / median **7.8%** / max 28.4%, **Spearman ρ=0.78**,
  champion 1/12, top-3 3/12. MAPE(global): 70b 29 / opt30b 17 / 8b 31 / qwen median 34%.
- **고친 버그 2개** (둘 다 champion이 아니라 정확도/물리 문제였음):
  1. *prefill 모델 ~8-10× 과대*: wall을 `T_pre + T_dec` 직렬합산 → prefill(tensor-core)과
     decode(HBM)는 자원 분리·부분중첩이므로 `max(T_pre,T_dec)+(1−ρ)·min` 블렌드로 교체
     (ρ=0.16 fit). + **prefill TFLOPS 2× 상향**(hw_params: BW 289→578, Ada 183→366;
     옛값은 additive 가정하 역산이라 저평가, free-fit이 1.98×로 독립 확인) +
     **prefill_ar_overlap=0.8**(async-TP가 cross-node AR 대부분 은닉). PLANNER_SPEC §5 갱신.
  2. *fit→prediction 단절*: `load_hardware()`가 `fitted_params.json`을 안 읽어 `--validate`/CLI가
     디폴트로 예측하던 버그 → 연결.
- **남은 gap (문서화됨, 미수정)**: (a) 70b TP8 prefill 여전히 ~2.4× 과소예측(388→540 개선했으나
  global ρ 하나로 decode/balanced(저ρ)와 TP8-prefill(고ρ) 동시충족 불가 → 70b prefill_heavy
  topology를 PP2로 오판, regret 15.7%; per-regime ρ 필요). (b) 8B 소형모델 overhead regime
  (LOMO 127%, 단일 step_floor 한계). (c) qwen TP8/PP4+ pathological cell(재측정 후보).
- fit 코드: `fit_planner.py`는 8 free param(7+ρ), prefill TFLOPS·AR-overlap은 hw_params 물리상수.

**Mistral 사전등록 완료** — `planner/mistral_prediction.json` 고정(sweep 데이터 나오기 전).
3 workload 모두 champion 예측 = **TP4PP2_layer_skew+12_56-32** (bal 805 / dec 906 / pre 587 tps).
`known_caveats` 3개 명시(skew workload-invariant, TP8-prefill 과소, 8B regime). **sweep 후 비교만 하면 됨.**

## 2026-06-15~16 세션 업데이트 (sweep 인프라 디버깅)
- **다운로드: 양 노드 완료** (head+worker 각 102 safetensors, /data/esca/.cache/huggingface/hub).
  ⚠️ Mistral repo는 `model-*`(표준 HF, 51) + `consolidated-*`(Mistral native, 51) **두 포맷 공존** → 490GB(디스크 87%). vllm는 `model.safetensors.index.json`(표준) 사용. consolidated-*는 정리해도 됨(245GB 회수).
- **Ray 재기동 절차 검증됨**: head `ray start --head --node-ip-address=10.20.0.30 --port=6379`(vllm_main/bin/ray, NCCL/GLOO_SOCKET_IFNAME=ibp3s0), worker SSH `ray start --address=10.20.0.30:6379 --node-ip-address=10.20.0.28 --num-gpus=4`(vllm_new/bin/ray, iface=ibp34s0). `ray status`로 head4+worker4=8 확인.
- **★ Mistral 로드 3시간+ stall 버그 = 원인규명+수정완료**: vllm `transformers_utils/config.py:get_safetensors_params_metadata`가 `--model <repo_id>`(로컬경로 아님)이면 HF Hub에서 파일별 헤더를 네트워크 fetch("Parse safetensors files" 진행바). **gated repo라 HF가 429 throttle → 233s/file, 51파일 ~3h**. 디스크/경합/타임아웃 전부 아님. **수정: sweep `_build_env`에 `HF_HUB_OFFLINE=1`+`TRANSFORMERS_OFFLINE=1` 추가** → 네트워크 호출 즉시실패(try_*가 무시)→로컬 fallback. 검증: 메타데이터 경로 3h→**2.03s**(1591 param). 70B가 됐던 건 그땐 throttle 없었을 뿐. wait_ready 타임아웃도 1200→4200s로 상향해둠.
- **하드룰 준수 확인**(스크립트): gpu_mem 0.85, enforce-eager 없음(CUDA graph ON), n_req=96, PP overlap=launcher `auto_configure` 4-env recipe.

## ★★ Mistral sweep + 사전등록 검증 완료 (2026-06-16) — 논문 헤드라인 결과
- **full 45셀 sweep 완료, 0 실패** (`results/hetero_4x4_mistral123b_full_20260616_211558/`, ~2h20m). HF_HUB_OFFLINE fix로 로드 정상(셀당 ~수분).
- **사전등록 예측 vs 측정 = champion 3/3, mean regret 0.0%, Spearman 0.836** (`planner/mistral_validation.json`).
  - 3 workload 전부 예측 챔피언 `TP4PP2_layer_skew+12_56-32`이 실측 챔피언과 일치(측정값 bal 993.8 / dec 1028.9 / pre 898.1 tps; runner-up skew+8과 0.5~1.9%차로 #1).
  - **완전 미지의 123B(캘리브 최대 70B의 ~1.8×)를 데이터 나오기 전 예측해 정타** → 플래너 일반성 최강 증거.
  - 사전등록 caveat "123B prefill이 70B처럼 TP8 우세면 빗나감"은 **적용 안 됨**: Mistral prefill은 PP2 우세(TP8 773 < PP2 898), 플래너가 PP2로 정타.
  - 단 magnitude MAPE 32%(과소예측 경향) — 순위는 정확하나 절대값은 거침(논문엔 regret/rank 중심으로 보고).

## 미완 ⚠️ (다음 세션)
1. (선택) consolidated-* 정리로 디스크 245GB 회수.
2. (선택) per-regime ρ 또는 async-TP 모델로 TP8-prefill gap 해소 → 70b prefill topology 정타.
3. ⚠️ **인프라**: head 노드가 16h 내 2회 reboot(주기적/불안정). reboot시 ray+잡 사망 → ray 재기동 절차(위 세션노트). 클러스터 매우 붐빔(다른 사용자들이 GPU 수시 점유), 남의 잡 불가침.

## 인프라 메모 (중요)
- **worker SSH 됨**: `ssh esca@10.20.0.28 '...'` 패스워드 없이. ray 재시작에 사용.
  단 head ray는 iface `ibp3s0`, worker는 `ibp34s0` (NCCL_SOCKET_IFNAME 틀리면 'NCCL invalid usage').
- **PP overlap recipe**: launcher.pp_overlap_config.auto_configure만 사용. VLLM_PP_OVERLAP /
  VLLM_PP_FAST_COMM 추가로 켜면 cross-node hang (→ [[pp-overlap-stale-env-bug]]).
- cluster 죽으면: head `ray start --head --node-ip-address=10.20.0.30 --port=6379`
  (NCCL_SOCKET_IFNAME=ibp3s0) → worker SSH로 `ray start --address=10.20.0.30:6379
  --node-ip-address=10.20.0.28 --num-gpus=4` (NCCL_SOCKET_IFNAME=ibp34s0).
- 하드룰: --enforce-eager 금지, n_req≤100(70B/Mistral은 96), gpu_mem 0.85.

## 2026-06-18 정리/일반화/백업 (서버 셧다운 대비)
- **백업**: `/data/esca/planner_backup_20260618.tar.gz` (코드+params+docs+figures+paper+최종results데이터+memory; 모델·로그 제외). scp로 외부 반출. ★셧다운 전 가져갈 것.
- **디스크**: Mistral consolidated-* blob 228GB 삭제(중복 native 포맷; model-* 51개 보존→로드 가능). 95%→81%.
- **results 정리**: keep-set 9개만 보존(모델별 full 1개 + 70b stock(122900) + 2x2(085153) + verify_pp_overlap + nsys). 나머지 62개 partial/debug 삭제.
- **코드 정리**: legacy v1(cost_model/planner/scorer/workload/gpu_library/network_library/model_spec/hetero_eval/run_eval/cli) + one-off diag/test + old-util → `planner/legacy_v1/`로 archive. `__init__.py`는 thin package marker로 교체.
- **★ 일반화**: 모델별 sweep 6개 → **`planner/hetero_sweep.py` 하나로 통합**(`legacy_v1/superseded_sweeps/`에 원본 보존). 모델 dims는 perf_planner.MODELS 재사용, config grid 자동생성, GPU 레이아웃 인자화. 사용:
  - `python planner/hetero_sweep.py --model 70b` (4+4 full, 3 workloads)
  - `python planner/hetero_sweep.py --model 70b --head-gpus 2 --worker-gpus 2 --workloads balanced`
  - `--dry-run`으로 config grid 검증 (전 모델 0 BAD 확인됨). 하드룰·HF offline·perf offline·pp_overlap recipe 내장.
  - ⚠️ dry-run만 검증됨(engine은 오늘 2x2 실행분과 동일). 실하드웨어 1-cell smoke로 run-validation 권장.
- **인프라 fix(이번 세션)**: HF gated 모델 로드 3h stall → `HF_HUB_OFFLINE=1`(build_env+perf both). 70b가 esca head 캐시에 없어 worker→head rsync + 타 사용자 캐시(읽기권한)서 tokenizer 복사. ⚠️ ray가 현재 **2+2** 상태 → 8-GPU 풀 sweep은 4+4로 재기동 필요.

## 파일 위치 (정리 후)
- 코어: `planner/{perf_planner,fit_planner,build_calibration,cluster_env,cluster_setup_4x4}.py` + `{PLANNER_SPEC.md, hw_params.json, calibration_data.csv, fitted_params.json, mistral_prediction.json, mistral_validation.json}`
- 일반 sweep: `planner/hetero_sweep.py` (구 모델별 6개는 `legacy_v1/superseded_sweeps/`)
- PP overlap 검증: `planner/{analyze_pp_overlap_trace,nsys_pp_overlap_*,verify_pp_overlap_torch_profiler}.py`
- plot: `planner/plot_*.py`  / archive: `planner/legacy_v1/`
- 측정 결과: `results/` (keep-set 9개)  / 그림: `figures/`  / 백업: `/data/esca/planner_backup_20260618.tar.gz`
- 메모리(세션 간 영속): `/data/esca/.claude/projects/-data-esca/memory/`
