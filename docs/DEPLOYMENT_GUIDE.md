# 🚀 Multi-Architecture Deployment Guide

## 🎯 Overview

This guide provides step-by-step instructions for deploying the multi-architecture LLM inference system to production. The system supports both Magistral and Mistral 7B architectures with automatic detection and handling.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.11+
- **Dependencies**: See `requirements.txt`
- **Memory**: 32GB+ RAM recommended for large models
- **Storage**: Fast SSD recommended for fragment loading
- **Network**: Low-latency connections for distributed inference

### Supported Architectures
| Architecture | Status | Dimensions |
|--------------|--------|------------|
| Magistral | ✅ Production Ready | dim=5120, hidden=32768 |
| Mistral 7B | ✅ Production Ready | dim=4096, hidden=14336 |

## 🚀 Deployment Steps

### Phase 1: Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy as np; print('NumPy:', np.__version__)"
```

### Phase 2: Configuration

#### Configuration File (`config/production.json`)

```json
{
  "models": {
    "magistral": {
      "fragments_dir": "models/Magistral-Small-2509-Q4_K_M_fragments",
      "architecture": "magistral",
      "use_binary": true,
      "use_compression": true,
      "collect_metrics": true
    },
    "mistral_7b": {
      "fragments_dir": "models/Mistral_7B_Instruct_v0_3_Q4_K_M_fragments",
      "architecture": "mistral_7b",
      "use_binary": true,
      "use_compression": true,
      "collect_metrics": true
    }
  },
  "network": {
    "timeout": 30.0,
    "retry_attempts": 3,
    "retry_backoff": 1.5
  },
  "monitoring": {
    "metrics_interval": 60,
    "health_check_interval": 300,
    "alert_thresholds": {
      "error_rate": 0.01,
      "latency": 1000,
      "memory_usage": 0.85
    }
  }
}
```

#### Environment Variables (`.env.production`)

```env
# Production settings
ENVIRONMENT=production
LOG_LEVEL=INFO

# Model paths
MAGISTRAL_FRAGMENTS=models/Magistral-Small-2509-Q4_K_M_fragments
MISTRAL_7B_FRAGMENTS=models/Mistral_7B_Instruct_v0_3_Q4_K_M_fragments

# Network settings
NETWORK_TIMEOUT=30.0
MAX_RETRIES=3

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
```

### Phase 3: Model Preparation

#### Verify Fragment Integrity

```bash
# Check Magistral fragments
python tools/verify_fragments.py \
  --fragments-dir models/Magistral-Small-2509-Q4_K_M_fragments \
  --expected-architecture magistral

# Check Mistral 7B fragments
python tools/verify_fragments.py \
  --fragments-dir models/Mistral_7B_Instruct_v0_3_Q4_K_M_fragments \
  --expected-architecture mistral_7b
```

**Expected Output:**
```
✅ Fragment count matches manifest
✅ All fragments readable
✅ Architecture detected: magistral/mistral_7b
✅ Tensor dimensions correct
✅ Checksums valid
```

### Phase 4: Deployment

#### Local Testing

```bash
# Test Magistral model
python app.py \
  --fragments-dir models/Magistral-Small-2509-Q4_K_M_fragments \
  --config config/production.json \
  --test-mode

# Test Mistral 7B model
python app.py \
  --fragments-dir models/Mistral_7B_Instruct_v0_3_Q4_K_M_fragments \
  --config config/production.json \
  --test-mode
```

#### Production Deployment

```bash
# Start production server
python app.py \
  --fragments-dir models/Magistral-Small-2509-Q4_K_M_fragments \
  --config config/production.json \
  --production-mode \
  --log-file production.log

# Start monitoring
python tools/monitor.py \
  --config config/production.json \
  --port 9090
```

### Phase 5: Verification

#### Health Checks

```bash
# Check system health
curl http://localhost:9090/health

# Expected response:
{
  "status": "healthy",
  "architecture": "magistral",
  "models_loaded": 40,
  "memory_usage": 0.65,
  "uptime": "2h30m",
  "error_rate": 0.001
}
```

#### Performance Testing

```bash
# Run performance tests
python tools/benchmark.py \
  --model magistral \
  --iterations 100 \
  --output benchmark_results.json

python tools/benchmark.py \
  --model mistral_7b \
  --iterations 100 \
  --output benchmark_results_mistral.json
```

**Expected Performance:**
- Tensor operations: <1ms
- Serialization: <0.5ms
- End-to-end inference: <10ms/token
- Memory usage: <80% of available RAM

## 🔧 Monitoring and Maintenance

### Key Metrics to Monitor

| Metric | Target | Alert Threshold |
|--------|-------|-----------------|
| Error Rate | <1% | >1% (warning), >5% (critical) |
| Latency | <10ms/token | >50ms (warning), >100ms (critical) |
| Memory Usage | <80% | >85% (warning), >90% (critical) |
| Throughput | >10 tokens/sec | <5 tokens/sec (warning) |
| Architecture Mismatches | 0 | >0 (critical) |

### Alerting Configuration

```yaml
# alerts.yml
alerts:
  - name: HighErrorRate
    condition: error_rate > 0.01
    severity: warning
    notification: "Error rate elevated: {error_rate}%"
    
  - name: CriticalErrorRate
    condition: error_rate > 0.05
    severity: critical
    notification: "CRITICAL: Error rate {error_rate}% - investigate immediately"
    
  - name: HighLatency
    condition: latency > 50
    severity: warning
    notification: "High latency detected: {latency}ms"
    
  - name: MemoryWarning
    condition: memory_usage > 0.85
    severity: warning
    notification: "Memory usage high: {memory_usage}%"
    
  - name: ArchitectureMismatch
    condition: architecture_mismatches > 0
    severity: critical
    notification: "CRITICAL: Architecture mismatch detected"
```

### Common Issues and Solutions

#### Issue: Architecture Mismatch Errors

**Symptoms:**
```
ValueError: Architecture mismatch during deserialization. Expected magistral, got mistral_7b
```

**Solutions:**
1. Verify all nodes use the same model architecture
2. Check configuration files for consistency
3. Restart nodes to ensure clean state
4. Verify fragment integrity

#### Issue: High Memory Usage

**Symptoms:**
- Memory usage >90%
- Slow response times
- Potential OOM kills

**Solutions:**
1. Reduce batch size
2. Enable fragment caching with `cache_raw=True`
3. Increase available memory
4. Optimize tensor operations

#### Issue: High Latency

**Symptoms:**
- Latency >100ms
- Timeout errors
- Slow inference

**Solutions:**
1. Check network connectivity
2. Verify load balancing
3. Optimize serialization format
4. Increase timeout settings

#### Issue: Serialization Errors

**Symptoms:**
- Serialization failures
- Data corruption
- Format errors

**Solutions:**
1. Verify zstandard installation
2. Check data types and shapes
3. Fallback to binary format
4. Update serialization libraries

## 📚 Architecture-Specific Notes

### Magistral Architecture

```json
{
  "dim": 5120,
  "hidden_dim": 32768,
  "n_layers": 40,
  "n_heads": 32,
  "n_kv_heads": 8,
  "vocab_size": 131072,
  "attention": {
    "q_dim": 2880,
    "k_dim": 2880,
    "v_dim": 4200,
    "output_dim": 2304
  },
  "ffn": {
    "gate_dim": 2880,
    "up_dim": 2880,
    "down_dim": 26880
  }
}
```

### Mistral 7B Architecture

```json
{
  "dim": 4096,
  "hidden_dim": 14336,
  "n_layers": 32,
  "n_heads": 32,
  "n_kv_heads": 8,
  "vocab_size": 32768,
  "attention": {
    "q_dim": 2304,
    "k_dim": 2304,
    "v_dim": 3360,
    "output_dim": 2304
  },
  "ffn": {
    "gate_dim": 2304,
    "up_dim": 2304,
    "down_dim": 11760
  }
}
```

## 🎯 Rollback Procedure

### Step 1: Identify Issue
- Check logs for errors
- Verify architecture detection
- Confirm tensor dimensions

### Step 2: Fallback to Single Architecture

```bash
# Fallback to Magistral only
python app.py \
  --fragments-dir models/Magistral-Small-2509-Q4_K_M_fragments \
  --single-architecture-mode \
  --emergency-fallback
```

### Step 3: Restore from Backup

```bash
# Restore previous version
git checkout tags/v1.0.0-stable
pip install -r requirements.txt

# Restart with known-good configuration
python app.py --safe-mode
```

### Step 4: Communicate
- Notify team of rollback
- Update status page
- Monitor restored system
- Plan fix for next deployment

## 📅 Maintenance Schedule

| Task | Frequency | Responsible |
|------|-----------|-------------|
| Health Checks | Daily | DevOps |
| Log Review | Daily | Engineering |
| Performance Review | Weekly | Performance Team |
| Fragment Verification | Weekly | Data Team |
| Dependency Updates | Monthly | Security Team |
| Architecture Review | Quarterly | Architecture Team |

## 🚀 Success Criteria

### Technical Success
- ✅ All architecture detection tests passing
- ✅ Both models producing valid outputs
- ✅ Performance within targets
- ✅ No critical bugs in production

### Operational Success
- ✅ Error rate <1% in first week
- ✅ 99.9% uptime over first month
- ✅ Positive user feedback
- ✅ Documentation complete and accessible

## 🎉 Celebration

**The team has successfully built and deployed a production-ready multi-architecture LLM inference system!** 🎉

This represents a major technical achievement that:
1. ✅ Adds full Mistral 7B support
2. ✅ Maintains backward compatibility
3. ✅ Provides extensibility for future architectures
4. ✅ Follows best practices for design and implementation
5. ✅ Is ready for production use

**Congratulations to the entire team!** 🎊

---

*This deployment guide provides comprehensive instructions for deploying the multi-architecture LLM inference system to production, including configuration, verification, monitoring, and troubleshooting.*