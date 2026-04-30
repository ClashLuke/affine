#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import bittensor as bt


@dataclass(frozen=True)
class WalletSpec:
    wallet_name: str
    hotkey_name: str
    cold_uri: str
    hot_uri: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap local dev-chain state for affine E2E")
    p.add_argument("--endpoint", default=os.getenv("SUBTENSOR_ENDPOINT", "ws://127.0.0.1:9944"))
    p.add_argument("--fallback", default=os.getenv("SUBTENSOR_FALLBACK", ""))
    p.add_argument("--wallet-path", default=os.path.expanduser("~/.bittensor/wallets"))
    p.add_argument("--netuid", type=int, default=None, help="Use existing netuid; create new if omitted")
    p.add_argument("--start-chain-cmd", default=os.getenv("AFFINE_DEVCHAIN_START_CMD", ""))
    p.add_argument("--wait-timeout", type=int, default=180)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--revision-a", default="main")
    p.add_argument("--revision-b", default="main")
    p.add_argument("--output", default=".e2e/dev_chain_state.json")
    p.add_argument("--validator-wallet", default="affine-e2e-validator")
    p.add_argument("--validator-hotkey", default="validator")
    p.add_argument("--miner-a-wallet", default="affine-e2e-miner-a")
    p.add_argument("--miner-a-hotkey", default="miner-a")
    p.add_argument("--miner-b-wallet", default="affine-e2e-miner-b")
    p.add_argument("--miner-b-hotkey", default="miner-b")
    p.add_argument("--cold-uri", default="//Alice")
    p.add_argument("--validator-hot-uri", default="//Alice//validator")
    p.add_argument("--miner-a-hot-uri", default="//Alice//miner_a")
    p.add_argument("--miner-b-hot-uri", default="//Alice//miner_b")
    return p.parse_args()


async def _wait_for_chain(endpoint: str, timeout: int) -> bt.AsyncSubtensor:
    deadline = time.time() + timeout
    while time.time() < deadline:
        sub = bt.AsyncSubtensor(endpoint)
        try:
            await sub.initialize()
            await sub.get_current_block()
            return sub
        except Exception:
            try:
                await sub.close()
            except Exception:
                pass
            await asyncio.sleep(2)
    raise TimeoutError(f"subtensor endpoint not reachable within {timeout}s: {endpoint}")


def _ensure_wallet(spec: WalletSpec, wallet_path: str) -> bt.Wallet:
    wallet = bt.Wallet(name=spec.wallet_name, hotkey=spec.hotkey_name, path=wallet_path)
    if not wallet.coldkey_file.exists_on_device():
        wallet.create_coldkey_from_uri(spec.cold_uri, use_password=False, overwrite=False, suppress=True)
    if not wallet.hotkey_file.exists_on_device():
        wallet.create_hotkey_from_uri(spec.hot_uri, use_password=False, overwrite=False, suppress=True)
    wallet.get_coldkey()
    wallet.get_hotkey()
    return wallet


async def _ensure_subnet(sub: bt.AsyncSubtensor, owner: bt.Wallet, target_netuid: int | None) -> int:
    existing = set(await sub.get_all_subnets_netuid())
    owner_hotkey = owner.hotkey.ss58_address
    if target_netuid is not None:
        if target_netuid not in existing:
            raise RuntimeError(f"netuid {target_netuid} does not exist on endpoint")
        return target_netuid

    owned = []
    for netuid in sorted(existing):
        try:
            if await sub.get_subnet_owner_hotkey(netuid) == owner_hotkey:
                owned.append(netuid)
        except Exception:
            continue
    if owned:
        return max(owned)

    await sub.register_subnet(wallet=owner, raise_error=False)

    await asyncio.sleep(2)
    netuids = sorted(set(await sub.get_all_subnets_netuid()))
    owned = []
    for netuid in netuids:
        try:
            if await sub.get_subnet_owner_hotkey(netuid) == owner_hotkey:
                owned.append(netuid)
        except Exception:
            continue
    if not owned:
        raise RuntimeError("unable to discover subnet owned by validator wallet")
    created = [n for n in owned if n not in existing]
    return max(created) if created else max(owned)


async def _ensure_registered(sub: bt.AsyncSubtensor, wallet: bt.Wallet, netuid: int, timeout: int = 180) -> None:
    hotkey = wallet.hotkey.ss58_address
    if await sub.is_hotkey_registered_on_subnet(hotkey, netuid):
        return
    await sub.burned_register(wallet=wallet, netuid=netuid, raise_error=False)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if await sub.is_hotkey_registered_on_subnet(hotkey, netuid):
            return
        await asyncio.sleep(2)
    raise TimeoutError(f"wallet hotkey failed to register on subnet {netuid}: {hotkey}")


async def _set_commitment(
    sub: bt.AsyncSubtensor, wallet: bt.Wallet, netuid: int, model: str, revision: str
) -> None:
    payload = json.dumps({"model": model, "revision": revision}, separators=(",", ":"))
    await sub.set_commitment(wallet=wallet, netuid=netuid, data=payload, mev_protection=False, raise_error=False)


async def _wait_commitments(
    sub: bt.AsyncSubtensor, netuid: int, expected_hotkeys: set[str], timeout: int = 300
) -> dict[str, str]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        revealed = await sub.get_all_revealed_commitments(netuid)
        if expected_hotkeys.issubset(set(revealed.keys())) and all(revealed[hk] for hk in expected_hotkeys):
            return {hk: max(v, key=lambda x: x[0])[1]
                    for hk, v in revealed.items() if v}
        raw = await sub.get_all_commitments(netuid)
        if expected_hotkeys.issubset(set(raw.keys())):
            return raw
        await asyncio.sleep(2)
    raise TimeoutError(f"commitments not visible for all expected hotkeys on netuid={netuid}")


def _start_chain_if_needed(cmd: str) -> subprocess.Popen | None:
    if not cmd:
        return None
    return subprocess.Popen(  # noqa: S602
        cmd,
        shell=True,
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def _async_main(args: argparse.Namespace) -> int:
    chain_proc = None
    endpoint = args.endpoint
    try:
        try:
            sub = await _wait_for_chain(endpoint, timeout=5)
        except Exception:
            chain_proc = _start_chain_if_needed(args.start_chain_cmd)
            sub = await _wait_for_chain(endpoint, timeout=args.wait_timeout)

        wallet_path = str(Path(args.wallet_path).expanduser())
        Path(wallet_path).mkdir(parents=True, exist_ok=True)
        validator_spec = WalletSpec(
            wallet_name=args.validator_wallet,
            hotkey_name=args.validator_hotkey,
            cold_uri=args.cold_uri,
            hot_uri=args.validator_hot_uri,
        )
        miner_a_spec = WalletSpec(
            wallet_name=args.miner_a_wallet,
            hotkey_name=args.miner_a_hotkey,
            cold_uri=args.cold_uri,
            hot_uri=args.miner_a_hot_uri,
        )
        miner_b_spec = WalletSpec(
            wallet_name=args.miner_b_wallet,
            hotkey_name=args.miner_b_hotkey,
            cold_uri=args.cold_uri,
            hot_uri=args.miner_b_hot_uri,
        )

        validator = _ensure_wallet(validator_spec, wallet_path)
        miner_a = _ensure_wallet(miner_a_spec, wallet_path)
        miner_b = _ensure_wallet(miner_b_spec, wallet_path)

        netuid = await _ensure_subnet(sub, validator, args.netuid)
        await _ensure_registered(sub, validator, netuid)
        await _ensure_registered(sub, miner_a, netuid)
        await _ensure_registered(sub, miner_b, netuid)

        await _set_commitment(sub, miner_a, netuid, args.model, args.revision_a)
        await _set_commitment(sub, miner_b, netuid, args.model, args.revision_b)
        commitments = await _wait_commitments(
            sub,
            netuid,
            {miner_a.hotkey.ss58_address, miner_b.hotkey.ss58_address},
        )

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "endpoint": endpoint,
            "fallback": args.fallback or "",
            "netuid": netuid,
            "wallet_path": wallet_path,
            "validator_wallet": args.validator_wallet,
            "validator_hotkey": args.validator_hotkey,
            "model": args.model,
            "miner_commitments": {
                miner_a.hotkey.ss58_address: {"model": args.model, "revision": args.revision_a},
                miner_b.hotkey.ss58_address: {"model": args.model, "revision": args.revision_b},
            },
            "revealed_hotkeys": sorted(commitments.keys()),
            "chain_started_by_script": bool(chain_proc),
            "chain_pid": chain_proc.pid if chain_proc else None,
        }
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

        print(f"Wrote bootstrap state: {out_path}")
        print("Export these for affine runtime:")
        print(f'export SUBTENSOR_ENDPOINT="{endpoint}"')
        if args.fallback:
            print(f'export SUBTENSOR_FALLBACK="{args.fallback}"')
        print(f'export NETUID="{netuid}"')
        print(f'export BT_WALLET_COLD="{args.validator_wallet}"')
        print(f'export BT_WALLET_HOT="{args.validator_hotkey}"')
        print(f'export BT_WALLET_PATH="{wallet_path}"')
        if chain_proc:
            print(f"# local chain pid: {chain_proc.pid}")
        return 0
    finally:
        try:
            await sub.close()  # type: ignore[name-defined]
        except Exception:
            pass


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
