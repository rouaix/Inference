# 🔧 Multi-Architecture Troubleshooting Guide

## 🎯 Introduction

This guide provides solutions to common issues that may arise when working with the multi-architecture LLM inference system. The system supports both Magistral and Mistral 7B architectures with automatic detection.

## 🚨 Common Issues and Solutions

### 1. Architecture Mismatch Errors

**Error Message:**
```
ValueError: Architecture mismatch during deserialization. Expected magistral, got mistral_7b
```

**Root Cause:**
- Different architectures being used between client and server
- Configuration mismatch
- Fragment corruption

**Solutions:**

1. **Verify Configuration:**
   ```bash
   # Check client configuration
   grep "architecture" config/client.json
   
   # Check server configuration
   grep "architecture" config/server.json
   ```

2. **Force Architecture:**
   ```python
   # Client side
   executor = RemoteLayerExecutor(
       "http://server:8000",
       architecture="magistral"  # Force specific architecture
   )
   ```

3. **Verify Fragments:**
   ```bash
   python tools/verify_fragments.py \
     --fragments-dir models/Magistral-Small-2509-Q4_K_M_fragments
   ```

4. **Restart Services:**
   ```bash
   # Restart client and server
   systemctl restart llm-inference-client
   systemctl restart llm-inference-server
   ```

**Prevention:**
- Use consistent configuration across all nodes
- Implement architecture validation in CI/CD
- Monitor for architecture mismatches

### 2. High Memory Usage

**Symptoms:**
- Memory usage >90%
- Slow response times
- Potential OOM kills
- `MemoryError` exceptions

**Root Causes:**
- Large batch sizes
- Memory leaks
- Fragment caching issues
- Insufficient system resources

**Solutions:**

1. **Reduce Batch Size:**
   ```python
   # Reduce batch size in configuration
   config = {
       "max_batch_size": 8,  # Reduced from 16
       "use_binary": True,
       "use_compression": True
   }
   ```

2. **Enable Fragment Caching:**
   ```python
   from distribution.local import LocalFragmentLoader
   
   loader = LocalFragmentLoader(
       fragments_dir,
       cache_raw=True  # Enable raw fragment caching
   )
   ```

3. **Optimize Tensor Operations:**
   ```python
   # Use more memory-efficient operations
   def optimized_forward(self, x):
       # Process in smaller chunks
       chunk_size = 4
       results = []
       for i in range(0, len(x), chunk_size):
           chunk = x[i:i+chunk_size]
           result = self._process_chunk(chunk)
           results.append(result)
       return np.concatenate(results, axis=0)
   ```

4. **Increase System Resources:**
   ```bash
   # Increase memory limits
   export MEMORY_LIMIT=32G
   
   # Use larger instance type
   gcloud compute instances resize instance-1 --machine-type=n2-standard-32
   ```

**Prevention:**
- Monitor memory usage continuously
- Set appropriate memory limits
- Implement automatic scaling
- Optimize fragment loading

### 3. High Latency

**Symptoms:**
- Latency >100ms
- Timeout errors
- Slow inference speeds
- User complaints about slowness

**Root Causes:**
- Network issues
- High server load
- Inefficient serialization
- Suboptimal configuration

**Solutions:**

1. **Check Network Connectivity:**
   ```bash
   # Test network latency
   ping server-address
   
   # Test network throughput
   iperf3 -c server-address
   ```

2. **Verify Server Load:**
   ```bash
   # Check CPU usage
   top
   
   # Check load average
   uptime
   
   # Check active connections
   netstat -an | grep ESTABLISHED
   ```

3. **Optimize Serialization:**
   ```python
   # Use binary format instead of JSON
   executor = RemoteLayerExecutor(
       "http://server:8000",
       use_binary=True,  # More efficient
       use_compression=True  # Reduce size
   )
   ```

4. **Adjust Timeout Settings:**
   ```python
   # Increase timeout in configuration
   config = {
       "timeout": 60.0,  # Increased from 30.0
       "retry_attempts": 3,
       "retry_backoff": 2.0
   }
   ```

**Prevention:**
- Monitor latency continuously
- Implement load balancing
- Optimize network configuration
- Use efficient serialization formats

### 4. Serialization Errors

**Symptoms:**
- Serialization failures
- Data corruption
- Format errors
- `SerializationError` exceptions

**Root Causes:**
- Missing zstandard library
- Incorrect data types
- Version mismatches
- Corrupted data

**Solutions:**

1. **Verify zstandard Installation:**
   ```bash
   pip install zstandard
   python -c "import zstandard; print('zstandard version:', zstandard.__version__)"
   ```

2. **Check Data Types:**
   ```python
   # Verify data types before serialization
   def validate_data(x):
       assert x.dtype == np.float32, f"Expected float32, got {x.dtype}"
       assert x.ndim == 2, f"Expected 2D array, got {x.ndim}D"
       return True
   ```

3. **Fallback to Binary Format:**
   ```python
   try:
       # Try compressed format
       return serialize_with_compression(data)
   except:
       # Fallback to binary format
       return serialize_binary(data)
   ```

4. **Update Libraries:**
   ```bash
   pip install --upgrade numpy zstandard
   ```

**Prevention:**
- Verify library versions
- Implement fallback mechanisms
- Validate data before serialization
- Monitor serialization success rates

### 5. Fragment Loading Issues

**Symptoms:**
- Missing fragments
- Fragment checksum mismatches
- Fragment format errors
- `FragmentLoadError` exceptions

**Root Causes:**
- Corrupted fragment files
- Incorrect fragment paths
- Version mismatches
- Permission issues

**Solutions:**

1. **Verify Fragment Integrity:**
   ```bash
   python tools/verify_fragments.py \
     --fragments-dir models/Magistral-Small-2509-Q4_K_M_fragments
   ```

2. **Check Fragment Paths:**
   ```python
   # Verify paths in configuration
   import os
   assert os.path.exists("models/Magistral-Small-2509-Q4_K_M_fragments")
   ```

3. **Re-download Fragments:**
   ```bash
   # Re-download fragments if corrupted
   python tools/download_fragments.py \
     --model magistral \
     --output-dir models/
   ```

4. **Check Permissions:**
   ```bash
   chmod -R 755 models/
   chown -R llm-user:llm-group models/
   ```

**Prevention:**
- Verify fragments before deployment
- Implement checksum validation
- Monitor fragment loading success
- Maintain backup fragments

### 6. Architecture Detection Failures

**Symptoms:**
- Architecture not detected
- Wrong architecture detected
- `UnsupportedArchitectureError`

**Root Causes:**
- Corrupted manifest files
- Missing manifest fields
- Unsupported architecture
- Version incompatibilities

**Solutions:**

1. **Verify Manifest:**
   ```bash
   cat models/Magistral-Small-2509-Q4_K_M_fragments/manifest.json
   ```

2. **Check Required Fields:**
   ```json
   {
     "dim": 5120,
     "hidden_dim": 32768,
     "vocab_size": 131072
   }
   ```

3. **Add Architecture Manually:**
   ```python
   executor = RemoteLayerExecutor(
       "http://server:8000",
       architecture="magistral"  # Manual override
   )
   ```

4. **Update Detection Logic:**
   ```python
   # Add new architecture to detection
   if dim == new_dim and hidden_dim == new_hidden:
       return ModelArchitecture.NEW_ARCHITECTURE
   ```

**Prevention:**
- Validate manifests before deployment
- Implement comprehensive error handling
- Monitor architecture detection success
- Maintain architecture compatibility list

## 🔧 Advanced Troubleshooting

### Network Diagnostics

```bash
# Check network connectivity
ping -c 4 server-address

# Check port availability
nc -zv server-address 8000

# Check DNS resolution
nslookup server-address

# Check firewall rules
sudo iptables -L
```

### Performance Profiling

```bash
# Profile Python code
python -m cProfile -o profile.out app.py

# Analyze profile
python tools/analyze_profile.py profile.out

# Memory profiling
python -m memory_profiler app.py
```

### Log Analysis

```bash
# Check application logs
tail -f /var/log/llm-inference/production.log

# Filter error logs
grep ERROR /var/log/llm-inference/production.log

# Analyze log patterns
python tools/analyze_logs.py --input production.log
```

### Configuration Debugging

```bash
# Check active configuration
python tools/show_config.py

# Validate configuration
python tools/validate_config.py config/production.json

# Compare configurations
python tools/compare_configs.py config/staging.json config/production.json
```

## 📚 Architecture-Specific Tips

### Magistral Architecture

**Common Issues:**
- Large memory footprint (32768 hidden dim)
- Longer initialization time
- Higher computational requirements

**Optimizations:**
```python
# Use smaller batch sizes
config = {"max_batch_size": 8}

# Enable compression
config = {"use_compression": True}

# Monitor memory usage
monitor.memory_usage(threshold=0.85)
```

### Mistral 7B Architecture

**Common Issues:**
- Different attention dimensions
- Vocabulary size limitations
- Grouped Query Attention (GQA) compatibility

**Optimizations:**
```python
# Verify GQA configuration
assert config.n_heads % config.n_kv_heads == 0

# Check vocabulary size
assert tokens < config.vocab_size

# Monitor attention operations
monitor.attention_operations()
```

## 🎯 Prevention Best Practices

### 1. Comprehensive Testing
- Test all architectures before deployment
- Verify backward compatibility
- Test edge cases and error conditions

### 2. Monitoring and Alerting
- Monitor key metrics continuously
- Set appropriate alert thresholds
- Implement comprehensive logging

### 3. Configuration Management
- Use version-controlled configurations
- Validate configurations automatically
- Document configuration changes

### 4. Performance Optimization
- Profile before optimizing
- Optimize hot paths first
- Balance speed and memory usage

### 5. Documentation
- Keep documentation updated
- Document architecture decisions
- Provide troubleshooting guides

## 🚀 Final Checklist

**Before Deployment:**
- [ ] Verify all fragments are intact
- [ ] Test both architectures locally
- [ ] Check configuration files
- [ ] Set up monitoring and alerting
- [ ] Document rollback procedure

**After Deployment:**
- [ ] Monitor system health
- [ ] Check error rates
- [ ] Verify performance
- [ ] Collect user feedback
- [ ] Update documentation

---

*This troubleshooting guide provides comprehensive solutions to common issues that may arise when working with the multi-architecture LLM inference system, helping ensure smooth operation and quick resolution of problems.*