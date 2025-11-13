docker rm -f zov_wan2.2_rope_alltoall_fusion_test_shape
docker rmi wan2.2_rocm7.0.2_pytorch_2.10.0.dev20251023_iris_0dfc460_2

docker build -f dockerfile_rocm_7.0.2 -t wan2.2_rocm7.0.2_pytorch_2.10.0.dev20251023_iris_0dfc460_2 .

docker run  -d \
        --network=host \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --shm-size=16G \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v /home/jialzhu:/home/jialzhu \
        --name zov_wan2.2_rope_alltoall_fusion_test_shape \
        -t wan2.2_rocm7.0.2_pytorch_2.10.0.dev20251023_iris_0dfc460_2

docker restart zov_wan2.2_rope_alltoall_fusion_test_shape
docker exec zov_wan2.2_rope_alltoall_fusion_test_shape git -C /app/Wan2.2 pull
sudo rocm-smi --setperfdeterminism 1000
docker exec -e ENABLE_TORCH_PROFILER=1 zov_wan2.2_rope_alltoall_fusion_test_shape bash /app/Wan2.2/tests/i2v.sh
sudo rocm-smi -r
# docker exec zov_wan2.2_rope_alltoall_fusion_test_shape  7z a -mx=9 wan2.2_triton_optimized_with_call_stack.7z /app/Wan2.2/trace_rank* 
# docker exec zov_wan2.2_rope_alltoall_fusion_test rm -rf /app/Wan2.2/trace_rank*
# docker exec zov_wan2.2_rope_alltoall_fusion_test mv /app/wan2.2_triton_optimized_with_call_stack.7z /app/home/jialzhu

# docker exec zov_wan2.2_rope_alltoall_fusion_test bash /app/Wan2.2/tests/i2v.sh
# docker exec zov_wan2.2_rope_alltoall_fusion_test  7z a -mx=9 wan2.2_original_with_call_stack.7z /app/Wan2.2/trace_rank* 
# docker exec zov_wan2.2_rope_alltoall_fusion_test rm -rf /app/Wan2.2/trace_rank*
# docker exec zov_wan2.2_rope_alltoall_fusion_test mv /app/wan2.2_original_with_call_stack.7z /app/home/jialzhu