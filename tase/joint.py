"""Utility to inspect MuJoCo Humanoid joint metadata.

Run this file to print joint count, names, qpos addresses and DOF addresses.
"""

import gymnasium as gym


def main():
    env = gym.make("Humanoid-v5")
    model = env.unwrapped.model

    print("=== JOINT COUNT ===")
    print(model.njnt)

    print("\n=== JOINT NAMES ===")
    for i in range(model.njnt):
        adr = model.name_jntadr[i]
        name = model.names[adr:].split(b'\x00', 1)[0].decode('utf-8')
        print(f"{i}: {name}")

    print("\n=== QPOS ADDRESSES ===")
    print(model.jnt_qposadr)

    print("\n=== DOF ADDRESSES ===")
    print(model.jnt_dofadr)


if __name__ == "__main__":
    main()
