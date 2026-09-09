ARG BASE_IMAGE
FROM ${BASE_IMAGE}
WORKDIR /workspace/x2_stairs
COPY requirements-smoke.txt /tmp/requirements-smoke.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-smoke.txt
COPY x2_stairs ./x2_stairs
COPY config ./config
# NOTE: x2_stairs/mujoco_env.py and mujoco_closed_loop.py import
# x2_stair_autonomy.{policy_contract,models} from the repo's src/ (reused, not
# duplicated, tensor-assembly logic). This build context is `training/`, so
# that directory isn't reachable here; if this container is ever used to run
# the local-MuJoCo tools (as opposed to smoke_train.py, the current CMD, which
# doesn't need them), build from the repo root instead with build context `.`
# and add `COPY src/x2_stair_autonomy ./src/x2_stair_autonomy`.
ENV PYTHONUNBUFFERED=1
CMD ["python", "x2_stairs/smoke_train.py", "--episodes", "32"]
