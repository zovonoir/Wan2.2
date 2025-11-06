docker rm -f zov_wan2.2_rope_alltoall_fusion_test
docker rmi wan2.2_rocm7.0.2_pytorch_2.10.0.dev20251023_iris_0dfc460

docker build -f dockerfile_rocm_7.0.2 -t wan2.2_rocm7.0.2_pytorch_2.10.0.dev20251023_iris_0dfc460 .

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
        --name zov_wan2.2_rope_alltoall_fusion_test \
        -t wan2.2_rocm7.0.2_pytorch_2.10.0.dev20251023_iris_0dfc460

docker restart zov_wan2.2_rope_alltoall_fusion_test
sleep 3
