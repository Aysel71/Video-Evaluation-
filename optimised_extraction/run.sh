cat > run_pyav_processing.sh << 'EOF'
#!/bin/bash
cd /mnt/public-datasets/a.mirzoeva/Video-MME
source ~/miniconda3/bin/activate video-mme-env
pip install av  # Устанавливаем PyAV, если еще не установлен
nohup python parallel_pyav_process.py > processing_pyav.log 2>&1 &
echo $! > processing_pyav_pid.txt
echo "PyAV processing started with PID $(cat processing_pyav_pid.txt)"
EOF

chmod +x run_pyav_processing.sh
