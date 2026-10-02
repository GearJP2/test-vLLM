# Thai LLM Serving Benchmark with vLLM

## 1. เป้าหมายและขอบเขต

Benchmark LLM ภาษาไทยขนาด 7B–14B บน GPU host เดียว เริ่มจาก Docker container เดียว แล้วเพิ่ม Kubernetes single pod เมื่อทดสอบ deployment จริง

คำถามหลัก: ข้อความไทยเดียวกันใช้กี่ tokens ในแต่ละ tokenizer, prefix caching/chunked prefill ส่งผลต่อ latency/throughput อย่างไร และแต่ละ configuration รองรับ load ได้เท่าไร

ระยะแรกวัด infrastructure และ tokenizer ไม่ใช่การจัดอันดับคุณภาพภาษาไทย หากเลือกโมเดลสำหรับ production ต้องเพิ่ม quality evaluation แยกต่างหาก

## 2. Local development และ GPU server

| งาน | เครื่องส่วนตัว | GPU server |
|---|---|---|
| Dataset, tokenizer analysis, runner/report development | เหมาะสม ไม่ต้องโหลด model weights | ทำได้ |
| Smoke test ด้วย mock streaming server หรือโมเดลเล็ก | เหมาะสม; GPU test ต้องมี GPU ที่รองรับ | ทำได้ |
| 7B–14B, long context, concurrent requests | ทำได้เมื่อ GPU/VRAM เพียงพอ | แนะนำสำหรับผลนำเสนอทีม |

ประมาณการเพื่อวางแผน: BF16/FP16 weights ใช้ราว 2 bytes/parameter หรือ 14–16 GB สำหรับ 7B–8B และ 28 GB สำหรับ 14B ยังไม่รวม KV cache, activations และ runtime overhead. GPU 24 GB อาจพอสำหรับ 7B–8B ที่ context/concurrency จำกัด; 48–80 GB มีพื้นที่สำหรับ 14B และ workload หนักมากขึ้น แต่ต้องตรวจ memory จริง ไม่รับประกันว่า matrix ทั้งหมดจะพอดี

เริ่ม GPU เดียวและ tensor parallelism = 1. Multi-GPU เป็นการทดลองแยก ห้ามรวม hardware ต่างกันเป็นการเปรียบเทียบ configuration เดียวกัน ใช้ runner บน GPU host หรือ network เดียวกัน; หากยิงจาก laptop ผ่านอินเทอร์เน็ต ให้แยกเป็น remote end-to-end results พร้อมระบุ network path

ก่อน benchmark ต้องบันทึก run manifest:

- GPU รุ่น/จำนวน/VRAM, CPU, RAM, OS/architecture, NVIDIA driver, CUDA/runtime และ container resource limits
- vLLM version และ image digest ที่แน่นอน ไม่ใช้ floating latest
- Model/tokenizer IDs และ commit revisions, chat template, dtype, quantization และ KV-cache dtype
- Effective server configuration: context limit, batch-token budget, max sequences, memory utilization, TP และ feature states
- Dataset hash, runner version/commit, seed, generation settings และ network placement

สิ่งที่ยังต้องเลือกเมื่อทราบเครื่องจริง: GPU, image/version, model revisions, resource limits และ latency SLO. เริ่มงาน dataset/runner ได้ก่อน แต่ห้ามสรุป production capacity โดยไม่มีบริบทนี้

## 3. Model shortlist

เริ่ม pipeline ด้วย Qwen2.5-7B-Instruct หนึ่งโมเดล แล้วขยายตามลำดับ:

| Model ID | บทบาท |
|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | Initial multilingual model |
| `typhoon-ai/llama3.1-typhoon2-8b-instruct` | Thai-focused model |
| `meta-llama/Llama-3.1-8B-Instruct` | Baseline; ตรวจ license/access ก่อน download |
| `Qwen/Qwen2.5-14B-Instruct` | Optional size comparison เมื่อ memory เพียงพอ |

ใช้ Instruct checkpoints และ template ของแต่ละโมเดล บันทึก template overhead แยกจากข้อความดิบ ตรวจ tokenizer จริง ไม่ถือว่า Thai fine-tuning ทำให้ tokenizer ต่างหรือดีกว่าโดยอัตโนมัติ

SeaLLM เป็น optional follow-up หลังเลือก exact checkpoint. ตัดรายการกำกวม “Llama 3.3 8B” ออกจาก matrix นี้

## 4. Dataset และ workloads

แยกการทดลองสองชุด:

1. **Same-text:** ข้อความไทย/task เดียวกันทุกโมเดล วัด token count และ practical task latency โดยยอมให้จำนวน tokens ต่างกัน รายงาน actual lengths และ truncation
2. **Controlled-token:** จัด lengths ตาม tokenizer ของแต่ละโมเดลเพื่อวัด serving ภายใต้ token load ที่กำหนด ไม่ถือว่าทุกโมเดลประมวลผลเนื้อหาเท่ากัน

| Workload | Input tokens สำหรับ controlled-token test | Output tokens |
|---|---|---|
| Short prompt | Buckets 100 และ 300 | 200 |
| Long context / RAG-like generation | Buckets 2,000 และ 8,000 | 500 |
| Shared prefix | System prefix 2,000 + unique query 100 | 200 |

RAG-like test ไม่รวม retrieval/embedding. นับ input หลังใช้ chat template; shared-prefix lengths เป็นเป้าหมายและต้องบันทึก actual lengths. ตรวจ context limit ว่ารองรับ input + output รวม template ก่อนรัน

- JSONL มี stable ID, workload, source/license และ task; ใช้ข้อมูลที่เผยแพร่ได้ พร้อม dataset hash/seed
- ใช้ข้อความไทยที่อ่านรู้เรื่อง ไม่ซ้ำทั้ง prompt โดยไม่ตั้งใจ
- Tokenizer report: tokens ต่อ Unicode code point ของข้อความดิบภายใต้ normalization ที่กำหนด; Thai/English token ratio ใช้คู่ประโยคที่ตรวจความหมายแล้ว ถ้าใช้ tokens/word ต้องระบุ Thai word segmenter/version
- Controlled-token mode ใช้ fixed output length และ ignore EOS หาก engine รองรับ ตรวจ actual length เสมอ
- Same-text natural-stop mode ใช้ output cap, temperature 0 และ sampling settings ที่ระบุชัดเจน รายงาน EOS/length finish reasons ไม่ถือว่า cap เท่ากับ output จริง

## 5. Configuration matrix

vLLM V1 อาจเปิด prefix caching และ chunked prefill โดย default จึงต้องกำหนด On/Off ชัดเจนและเก็บ resolved configuration

| Variant | Prefix caching | Chunked prefill |
|---|---|---|
| A | Off | Off |
| B | On | Off |
| C | Off | On |
| D | On | On |
| E (reference) | Engine default | Engine default |

สำหรับ A–D คง memory utilization, max sequences, context limit, dtype และ batch-token budget เท่ากัน เลือก budget ที่ valid เมื่อปิด chunked prefill ตาม pinned version (initial matrix ใช้ค่ามากกว่า context limit). ตรวจ supported combinations; ถ้าไม่รองรับให้รายงาน unsupported ไม่ fallback เงียบ ๆ

E ใช้ hardware/model/dtype/context เดียวกัน แต่ปล่อย tuning defaults ของ engine พร้อมบันทึกค่าจริง เป็น reference ไม่ใช่ causal comparison ของ feature เดียว

หลัง A–D จึง sweep batch-token budget เช่น 512, 2,048, 8,192 ใน configuration ที่เปิด chunked prefill และ valid กับ scheduler settings. Memory utilization 0.90 เป็น initial candidate เมื่อ memory check ผ่านและคงเท่ากันระหว่าง A–D; sweep memory แยกต่างหาก ไม่เรียก “Full Tuned” ก่อนมีผลทดลอง

## 6. Load protocol และ cache isolation

- Initial closed-loop concurrency sweep: 1, 4, 8, 16, 32 ส่ง request ใหม่เมื่อมี slot ว่าง และตรวจว่า client ไม่เป็น bottleneck
- เริ่ม smoke test: 1 model × short/100-token workload × concurrency 1 แล้วขยาย matrix ทีละส่วน
- Pilot ต่อ cell: 20 unmeasured warm-up requests, 200 measured requests และอย่างน้อย 3 repetitions รายงาน duration/sample count จริง เพิ่ม samples เมื่อผลยังไม่นิ่งหรือจะอ้าง tail latency
- Initial per-request timeout 120 seconds (configurable); ไม่ retry เงียบ ๆ นับ failed/timed-out/cancelled requests หยุด cell ที่ OOM และบันทึกสถานะ
- สลับลำดับ variants ด้วย seed ที่บันทึกไว้ ตรวจไม่มี workload อื่นแย่ง GPU แยก startup/compilation ออกจาก measured window
- Warm up ด้วย unrelated prompts แล้ว reset prefix cache โดยวิธีที่ version รองรับ ตรวจ reset สำเร็จ หรือ restart แล้ว warm up ใหม่ด้วย unrelated prompts
- **Cold-start shared-prefix run:** เริ่มโดยไม่มี target prefix cached แล้วปล่อย cache อุ่นตาม requests รายงาน first request แยก; ผลรวมไม่ใช่ all-miss workload
- **Warm shared-prefix run:** หลัง reset ให้ preload target prefix จน request จบก่อนวัด ใช้ prefix token IDs เดิมภายในโมเดลเดียวกันและ unique queries ตัด preload ออกจาก measured window
- เก็บ metrics snapshot หลัง warm-up/preload และหลัง drain requests จบ พร้อม periodic scrape เพื่อเห็น peak usage
- Optional open-loop request-rate sweep: แยก offered rate, achieved rate และ concurrency cap เพื่อวิเคราะห์ queueing/saturation

## 7. Metrics และ artifacts

ใช้ streaming client สำหรับ user-visible latency และ /metrics สำหรับ server diagnostics. พิจารณาใช้ `vllm bench serve` ของ version ที่ pin โดยให้ `benchmark_runner.py` orchestrate dataset/matrix/results

| Metric | Definition / reporting |
|---|---|
| TTFT | เวลาจากส่ง request จนได้ content แรก รวม queue/network ไม่ใช่ prefill อย่างเดียว |
| TPOT | (last-token time − first-token time) / (output tokens − 1); N/A เมื่อ output ≤1 token |
| Inter-token latency | ช่องว่างระหว่าง streamed outputs; chunk อาจมีหลาย tokens จึงไม่เท่ากับ TPOT เสมอ |
| End-to-end latency | เวลาจากส่ง request จน request จบ |
| Throughput | Successful completed requests/s และ actual output tokens จาก successful requests/s ใน measured window รวม drain |
| Reliability | attempted/succeeded/failed/timed-out, finish reasons, OOM และ preemptions |
| Prefix cache hit ratio | Delta hits / delta queries ภายใน measured window ตามหน่วย exporter; denominator 0 เป็น N/A |
| KV cache occupancy | ค่าเฉลี่ย/สูงสุดของ used cache blocks ไม่ใช่ total GPU memory |

รายงาน p50/p95 ของ TTFT, TPOT และ end-to-end latency พร้อมผลแต่ละ repetition และความแปรปรวนระหว่างรอบ. ตรวจชื่อจาก endpoint จริง: current docs ใช้ `vllm:kv_cache_usage_perc` และ prefix-cache query/hit counters; exporter อาจมี `_total` suffix. ตรวจ HELP/TYPE/labels รวม counters อย่างสอดคล้องกัน และแปลงหน่วย 0–1 เป็น percent อย่างชัดเจน

หากมี SLO ให้รายงาน goodput (successful requests/s ที่ผ่าน latency targets ทั้งหมด). หากยังไม่มี ให้รายงาน trade-offs ไม่ประกาศผู้ชนะหนึ่งเดียว

ถ้ารายงานต้นทุน ให้ระบุราคา server ต่อชั่วโมงและต้นทุนต่อ 1,000 successful requests ของ workload เดียวกัน แยก startup/idle/download cost; token count อย่างเดียวไม่ใช่ค่าใช้จ่ายจริง

เก็บ run manifest, dataset/hash, raw per-request JSONL, metrics snapshots/time series, server logs, CSV และ Markdown report พร้อม reproducible commands. ห้ามใส่ access tokens ใน artifacts

## 8. Implementation checklist และ acceptance criteria

ติดตามงานใน [GitHub issue #1](https://github.com/GearJP2/test-vLLM/issues/1)

### Current local scaffold

The repository now includes a starting configuration at `config/benchmark_matrix.json`, a versioned dataset seed at `data/thai_workloads.jsonl`, and a dependency-free structural check:

```bash
python3 scripts/preflight.py --allow-placeholders
```

This local check is intentionally not a GPU benchmark. Before a real GPU run, replace all placeholder documents with licensed Thai text, make the shared prefix byte-identical across its requests, confirm the pinned model revision, and run `python3 scripts/preflight.py` without the flag.

### EC2 Docker smoke test

Run the server and the benchmark runner on the same EC2 host. The API binds only to `127.0.0.1`, so it is not exposed publicly by the compose file.

```bash
cp .env.example .env
# Edit .env: use an immutable vLLM image digest, a model commit SHA, and one variant's VLLM_ARGS.
chmod +x scripts/ec2_preflight.sh
./scripts/ec2_preflight.sh
docker compose config
docker compose up -d
curl --fail http://127.0.0.1:8000/health
python3 scripts/benchmark_runner.py --variant D --workload short --concurrency 1 --requests 2 --warmup-requests 1
docker compose logs --tail=100 vllm
docker compose down
```

The smoke command is deliberately small. It proves that streaming, metrics scraping and result artifacts work; it is not a benchmark result. Do not begin the A–E matrix until all dataset placeholders have been replaced and `python3 scripts/preflight.py` passes without `--allow-placeholders`.

The container runs as root to access its model cache. If a host-side runner cannot write to `results/` after the first container start, repair the bind-mount ownership once:

```bash
sudo chown -R "$USER":"$USER" results
```

After pinning the model revision and replacing the dataset placeholders, generate a token-length report inside the vLLM container:

```bash
docker compose exec vllm python /workspace/scripts/tokenizer_report.py
```

The report is written to `results/tokenizer-report.json`. Review actual post-template token counts before selecting the final prompt documents or making any tokenizer-efficiency claim.

For the initial serving pilot, generate an authored synthetic Thai corpus on the GPU host. The script uses the pinned tokenizer and writes JSONL through standard output, so the resulting file remains owned by the host user:

```bash
docker compose exec -T vllm python /workspace/scripts/build_thai_workloads.py \
  > results/thai-workloads-qwen.jsonl

docker compose exec -T vllm python /workspace/scripts/tokenizer_report.py \
  --dataset /workspace/results/thai-workloads-qwen.jsonl \
  --output /workspace/results/tokenizer-report-qwen.json
```

Review the report and preserve its hash with the results. The generated corpus is suitable for infrastructure testing only; replace it with representative licensed production-like text before making a production capacity recommendation.

- [ ] Preflight: เลือก GPU/image/revisions ตรวจ model access, memory budget และ supported feature combinations
- [ ] Pinned Docker deployment พร้อม health check และ start/stop commands
- [ ] Thai JSONL datasets, tokenizer report และ lengths/template validation
- [ ] Runner/config สำหรับ A–E, concurrency sweep, warm-up, cache isolation, timeout และ raw results
- [ ] Client latency และ server metrics ที่ใช้ measured window เดียวกัน
- [ ] Deterministic streaming/metric calculation validation และ GPU smoke test หนึ่ง cell
- [ ] Pilot หนึ่งโมเดลครบ A–E และ 3 workloads ก่อนขยาย models; บันทึก failed/unsupported cells
- [ ] CSV/Markdown report: hardware, variance, configuration trade-offs และ limitations
- [ ] Optional: SLO/goodput, quality/cost evaluation, budget sweep, 14B และ Kubernetes single-pod validation

เสร็จเมื่อรันซ้ำได้จาก documented commands บน hardware ที่ระบุ, results ผูกกับ manifest/dataset, feature states ตรวจสอบได้, warm-up ไม่ปะปนกับ measured results และ report ไม่ซ่อน failures หรืออ้างเกินขอบเขตการทดสอบ

## References

- [vLLM GPU requirements](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [vLLM tuning and chunked-prefill constraints](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [vLLM metric definitions](https://docs.vllm.ai/en/latest/design/metrics/)
- [vLLM benchmark CLI](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md)
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Typhoon2-8B-Instruct](https://huggingface.co/typhoon-ai/llama3.1-typhoon2-8b-instruct)
- [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

References ชี้ latest docs เพื่อค้นข้อมูล; ตอน implement ต้องยึด documentation/help ของ pinned version และบันทึกความต่าง
