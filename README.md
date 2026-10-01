# Role & Goal
คุณคือ Senior AI/ML Infrastructure Engineer หน้าที่ของคุณคือช่วยฉันวางระบบและ Benchmark ประสิทธิภาพการให้บริการ LLM ภาษาไทยบน Single Pod ด้วย vLLM ก่อนนำผลลัพธ์ไปนำเสนอในทีม

---

## 1. Environment & Hardware Setup
- **Hardware:** NVIDIA GPU ( single pod / single GPU หรือ Multi-GPU Tensor Parallelism เช่น 1x A100/H100 หรือ DGX Spark ตามสภาพแวดล้อมจริง)
- **Serving Engine:** vLLM (Latest Version)
- **Orchestrator:** Kubernetes (Single Pod Deployment) หรือ Docker Container

---

## 2. Target Models for Testing
เลือกทดสอบโมเดล 7B - 14B ที่รองรับภาษาไทยได้ดี เพื่อเปรียบเทียบเรื่อง Tokenizer Efficiency:
1. **SeaLLM / Typhoon 2 (7B/8B/14B):** โมเดลที่ Optimize Tokenizer สำหรับภาษาไทยมาโดยเฉพาะ
2. **Qwen2.5 (7B / 14B):** โมเดล Multilingual ที่มี Vocabulary Size ใหญ่ (กินโทเคนไทยน้อย)
3. **Llama 3.1 / 3.3 (8B):** Base/Instruct Model มาตรฐานเพื่อใช้เป็น Baseline

---

## 3. Test Dataset & Input Scenarios
สร้าง Dataset ภาษาไทย 3 สไตล์เพื่อวัดผลตาม Behavior ของ Workload จริง:
1. **Short Prompt / High Concurrency:** คำถามทั่วไปสั้นๆ (Input 100-300 tokens, Output 200 tokens)
2. **Long Context / RAG Workflow:** เอกสารภาษาไทยยาวๆ เช่น สรุปรายงาน/เอกสารราชการ (Input 2,000 - 8,000 tokens, Output 500 tokens)
3. **Shared Prefix Workload:** ใช้ System Prompt ภาษาไทยยาวๆ เดียวกันทุก Request เพื่อทดสอบ Prefix Caching (Input: System Prompt 2,000 tokens + User Query 100 tokens)

---

## 4. Test Matrix & Configuration Variants
ทำการรัน Benchmark เปรียบเทียบ vLLM ตาม Flag ต่างๆ ดังนี้:
- **Variant A (Baseline):** vLLM Default Settings
- **Variant B (Prefix Caching):** `--enable-prefix-caching`
- **Variant C (Chunked Prefill):** `--enable-chunked-prefill` `--max-num-batched-tokens 512`
- **Variant D (Full Tuned):** `--enable-prefix-caching` + `--enable-chunked-prefill` + `--gpu-memory-utilization 0.90`

---

## 5. Evaluation Metrics
ดึงข้อมูลสถิติผ่าน vLLM Prometheus Metrics (`/metrics`) และ `benchmark_serving.py`:

1. **Tokenizer & Cost Metrics:**
   - **Compression Ratio / Tokens per Word:** จำนวน Token ภาษาไทยที่ได้เทียบกับภาษาอังกฤษในประโยคความหมายเดียวกัน
2. **Latency Metrics:**
   - **Time to First Token (TTFT):** ความเร็วในการเริ่มสร้างคำแรก (วัดผลกระทบจาก Prefill Phase ของภาษาไทย)
   - **Time Per Output Token (TPOT) / Inter-token Latency:** ความเร็วในการเจนคำถัดๆ ไป
3. **Throughput & Efficiency Metrics:**
   - **Request Throughput (req/s):** จำนวน Request ที่รับได้ต่อวินาที
   - **Token Throughput (tokens/s):** จำนวน Output Token ที่ผลิตได้ต่อวินาที
   - **Cache Hit Rate (`vllm:gpu_prefix_cache_hit_rate`):** อัตราการ Hit ของ Prefix Caching
   - **KV Cache Memory Usage (`vllm:gpu_cache_usage_perc`):** เปอร์เซ็นต์การดึง Memory ไปใช้

---

## 6. Execution Steps for Codex
โปรดช่วยฉันดำเนินการตามลำดับดังนี้:
1. เขียนสคริปต์ Dockerfile หรือ Kubernetes Deployment YAML สำหรับเตรียม vLLM Server
2. เขียนสคริปต์ Python (`benchmark_runner.py`) สำหรับยิง Request ภาษาไทยจำลองแบบต่างๆ ตาม Dataset ข้อ 3
3. ดึงค่า Metrics (TTFT, TPOT, Throughput, Cache Hit Rate) มารวบรวมเป็น Markdown Table เพื่อสรุปผลการเปรียบเทียบแต่ละ Variant