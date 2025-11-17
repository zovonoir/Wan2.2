docker rm -f wan2.2_testing_container
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
        --name wan2.2_testing_container \
        -t wan2.2_rocm7.0.2_pytorch_2.10.0.dev20251023_iris_0dfc460_2

docker restart wan2.2_testing_container
docker exec wan2.2_testing_container git -C /app/Wan2.2 pull
docker exec wan2.2_testing_container git -C /app/Wan2.2 checkout i2v_release/triton_kernel_optimized1
docker exec -e ENABLE_TIMING=1 wan2.2_testing_container bash /app/Wan2.2/tests/i2v.sh