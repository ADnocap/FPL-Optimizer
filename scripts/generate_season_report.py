#!/usr/bin/env python3
"""Generate a detailed GW-by-GW markdown report of the MILP optimizer run."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fpl_optimizer.data.downloader import DEFAULT_DATA_DIR
from fpl_optimizer.data.loader import SeasonDataLoader
from fpl_optimizer.engine.engine import FPLGameEngine
from fpl_optimizer.engine.state import ChipState, EngineAction, GameState, PlayerSlot, Squad
from fpl_optimizer.optimizer.squad_selection import select_squad
from fpl_optimizer.optimizer.transfer_optimizer import optimize_transfers
from fpl_optimizer.optimizer.types import build_candidate_pool, to_engine_action
from fpl_optimizer.prediction.integration import PredictionIntegrator
from fpl_optimizer.utils.constants import INITIAL_FREE_TRANSFERS, STARTING_BUDGET, Position


SEASON = "2024-25"
MAX_XFERS = 1  # human-realistic constraint


def build_name_map(loader: SeasonDataLoader) -> dict[int, str]:
    """Build element_id -> player name from merged_gw."""
    name_map: dict[int, str] = {}
    for _, row in loader._merged_gw[["element", "name"]].drop_duplicates("element").iterrows():
        name_map[int(row["element"])] = str(row["name"])
    return name_map


def build_team_name_map(loader: SeasonDataLoader) -> dict[int, str]:
    """Build team_id -> team short_name from teams.csv."""
    team_map: dict[int, str] = {}
    if not loader._teams.empty:
        for _, row in loader._teams.iterrows():
            team_map[int(row["id"])] = str(row.get("short_name", row.get("name", "?")))
    return team_map


def pos_abbr(pos: Position) -> str:
    return pos.name


def formation_str(players: list[PlayerSlot], lineup_indices: list[int]) -> str:
    lineup_players = [players[i] for i in lineup_indices]
    c = Counter(p.position for p in lineup_players)
    return f"{c.get(Position.DEF, 0)}-{c.get(Position.MID, 0)}-{c.get(Position.FWD, 0)}"


def main():
    data_dir = DEFAULT_DATA_DIR
    pred_data_dir = data_dir.parent if data_dir.name == "raw" else data_dir

    print(f"Loading {SEASON} data...")
    loader = SeasonDataLoader(SEASON, data_dir)
    engine = FPLGameEngine(loader)
    name_map = build_name_map(loader)
    team_names = build_team_name_map(loader)

    # Load predictions
    print("Loading prediction model...")
    model_dir = Path("models/point_predictor")
    integrator = PredictionIntegrator.from_model(model_dir, pred_data_dir, SEASON)

    def pred_fn(gw):
        all_eids = loader.get_all_element_ids(gw)
        return {eid: integrator.get_predicted_points(eid, gw) for eid in all_eids}

    num_gws = min(loader.get_num_gameweeks(), 38)

    # --- GW1: initial squad ---
    print("Selecting GW1 squad...")
    pp1 = pred_fn(1)
    candidates = build_candidate_pool(loader, 1, pp1)
    squad_result = select_squad(candidates, budget=STARTING_BUDGET)

    players = []
    for eid in squad_result.squad_element_ids:
        pos = loader.get_player_position(eid)
        price = loader.get_player_price(eid, 1)
        if pos and price > 0:
            players.append(PlayerSlot(element_id=eid, position=pos,
                                      purchase_price=price, selling_price=price))

    eid_to_idx = {p.element_id: i for i, p in enumerate(players)}
    lineup = [eid_to_idx[eid] for eid in squad_result.lineup_element_ids if eid in eid_to_idx]
    bench = [eid_to_idx[eid] for eid in squad_result.bench_element_ids if eid in eid_to_idx]
    captain_idx = eid_to_idx.get(squad_result.captain_id, 0)
    vice_idx = eid_to_idx.get(squad_result.vice_captain_id, 1)

    squad = Squad(players=players, lineup=lineup, bench=bench,
                  captain_idx=captain_idx, vice_captain_idx=vice_idx)
    state = GameState(squad=squad, bank=STARTING_BUDGET - squad_result.total_cost,
                      free_transfers=INITIAL_FREE_TRANSFERS, chips=ChipState(),
                      current_gw=1, total_points=0)

    # Collect GW data
    gw_logs = []

    for gw in range(1, num_gws + 1):
        print(f"  Processing GW{gw}...", end="\r")
        pp = pred_fn(gw)
        candidates = build_candidate_pool(loader, gw, pp)

        transfers_in_eids = []
        transfers_out_eids = []
        chip_used = None

        if not candidates:
            action = EngineAction()
        elif gw == 1:
            action = to_engine_action(squad_result)
            action.transfers_in = []
            action.transfers_out = []
        else:
            try:
                opt_result = optimize_transfers(state, candidates, max_transfers=MAX_XFERS)
                action = to_engine_action(opt_result)
                transfers_in_eids = list(opt_result.transfers_in)
                transfers_out_eids = list(opt_result.transfers_out)
                chip_used = opt_result.chip
            except RuntimeError:
                action = EngineAction()

        # Capture pre-step state
        pre_squad = state.squad
        pre_bank = state.bank
        pre_ft = state.free_transfers
        pre_lineup_eids = [pre_squad.players[i].element_id for i in pre_squad.lineup]
        pre_bench_eids = [pre_squad.players[i].element_id for i in pre_squad.bench]
        pre_captain_eid = pre_squad.players[pre_squad.captain_idx].element_id
        pre_vice_eid = pre_squad.players[pre_squad.vice_captain_idx].element_id
        pre_formation = formation_str(pre_squad.players, pre_squad.lineup)

        try:
            new_state, step_result = engine.step(state, action)
        except ValueError:
            action = EngineAction()
            new_state, step_result = engine.step(state, action)

        # After transfers but before engine step, figure out the actual lineup
        # Use post-step state for the lineup that actually played
        post_squad = new_state.squad
        post_lineup_eids = [post_squad.players[i].element_id for i in post_squad.lineup]
        post_bench_eids = [post_squad.players[i].element_id for i in post_squad.bench]
        post_captain_eid = post_squad.players[post_squad.captain_idx].element_id
        post_vice_eid = post_squad.players[post_squad.vice_captain_idx].element_id
        post_formation = formation_str(post_squad.players, post_squad.lineup)

        # Get individual player points for this GW
        player_pts = {}
        for p in post_squad.players:
            data = loader.get_player_gw(p.element_id, gw)
            player_pts[p.element_id] = int(data["total_points"]) if data else 0

        # If GW1, use the squad_result lineup info
        if gw == 1:
            lineup_eids = [eid for eid in squad_result.lineup_element_ids if eid in eid_to_idx]
            bench_eids = [eid for eid in squad_result.bench_element_ids if eid in eid_to_idx]
            captain_eid = squad_result.captain_id
            vice_eid = squad_result.vice_captain_id
            form = formation_str(players, [eid_to_idx[e] for e in lineup_eids])
        else:
            # Use the action's lineup (what the optimizer chose)
            if action.lineup:
                # Map back to the pre-transfer squad
                lineup_eids = list(action.lineup)
                bench_eids = list(action.bench) if action.bench else []
                captain_eid = action.captain or post_captain_eid
                vice_eid = action.vice_captain or post_vice_eid
            else:
                lineup_eids = post_lineup_eids
                bench_eids = post_bench_eids
                captain_eid = post_captain_eid
                vice_eid = post_vice_eid
            form = post_formation

        gw_logs.append({
            "gw": gw,
            "lineup": lineup_eids,
            "bench": bench_eids,
            "captain": captain_eid,
            "vice_captain": vice_eid,
            "formation": form,
            "transfers_in": transfers_in_eids,
            "transfers_out": transfers_out_eids,
            "chip": chip_used,
            "bank": pre_bank,
            "free_transfers": pre_ft,
            "gross_points": step_result.gw_points,
            "hit_cost": step_result.hit_cost,
            "net_points": step_result.net_points,
            "captain_points": step_result.captain_points,
            "bench_points": step_result.bench_points,
            "auto_subs": step_result.auto_subs,
            "captain_failover": step_result.captain_failover,
            "cumulative": new_state.total_points,
            "player_pts": player_pts,
            "post_bank": new_state.bank,
        })

        state = new_state

    print("\nGenerating report...")

    # --- Build markdown ---
    lines = []
    lines.append(f"# MILP Optimizer Season Report — {SEASON}")
    lines.append("")
    lines.append(f"**Strategy**: LightGBM predictions + MILP optimizer, max {MAX_XFERS} transfer/GW")
    lines.append(f"**Final Score**: {state.total_points} net points")
    total_gross = sum(g["gross_points"] for g in gw_logs)
    total_hits = sum(g["hit_cost"] for g in gw_logs)
    total_xfers = sum(len(g["transfers_out"]) for g in gw_logs)
    lines.append(f"**Gross Points**: {total_gross}")
    lines.append(f"**Total Hits**: {total_hits} pts ({total_hits // 4} extra transfers)")
    lines.append(f"**Total Transfers**: {total_xfers}")
    chips_used = [g["chip"] for g in gw_logs if g["chip"]]
    lines.append(f"**Chips Used**: {', '.join(chips_used) if chips_used else 'None'}")
    lines.append("")

    # Summary table
    lines.append("## Season Summary")
    lines.append("")
    lines.append("| GW | Formation | Gross | Hits | Net | Cumul | Captain | Transfers | Chip | Bank |")
    lines.append("|---:|:---------:|------:|-----:|----:|------:|:--------|:----------|:-----|-----:|")
    for g in gw_logs:
        capt_name = name_map.get(g["captain"], str(g["captain"]))
        xfer_str = ""
        if g["transfers_in"]:
            ins = [name_map.get(e, str(e)) for e in g["transfers_in"]]
            outs = [name_map.get(e, str(e)) for e in g["transfers_out"]]
            xfer_str = ", ".join(f"{o}→{i}" for o, i in zip(outs, ins))
        chip_str = g["chip"] or ""
        bank_str = f"£{g['post_bank'] / 10:.1f}m"
        lines.append(
            f"| {g['gw']} | {g['formation']} | {g['gross_points']} | "
            f"{g['hit_cost']} | {g['net_points']} | {g['cumulative']} | "
            f"{capt_name} | {xfer_str} | {chip_str} | {bank_str} |"
        )
    lines.append("")

    # Detailed per-GW breakdown
    lines.append("---")
    lines.append("")
    lines.append("## Gameweek Details")
    lines.append("")

    for g in gw_logs:
        lines.append(f"### GW{g['gw']}")
        lines.append("")
        lines.append(f"**Formation**: {g['formation']} | "
                      f"**Free Transfers**: {g['free_transfers']} | "
                      f"**Bank**: £{g['bank'] / 10:.1f}m | "
                      f"**Chip**: {g['chip'] or 'None'}")
        lines.append("")

        # Transfers
        if g["transfers_in"]:
            lines.append("**Transfers**:")
            for out_eid, in_eid in zip(g["transfers_out"], g["transfers_in"]):
                out_name = name_map.get(out_eid, str(out_eid))
                in_name = name_map.get(in_eid, str(in_eid))
                lines.append(f"- OUT: {out_name} → IN: {in_name}")
            if g["hit_cost"] > 0:
                lines.append(f"- **Hit cost**: -{g['hit_cost']} pts")
            lines.append("")

        # Starting XI
        lines.append("**Starting XI**:")
        lines.append("")
        lines.append("| # | Player | Pos | Pts | Role |")
        lines.append("|--:|:-------|:---:|----:|:-----|")
        for i, eid in enumerate(g["lineup"], 1):
            pname = name_map.get(eid, str(eid))
            pos = loader.get_player_position(eid)
            pos_str = pos.name if pos else "?"
            pts = g["player_pts"].get(eid, 0)
            role = ""
            if eid == g["captain"]:
                role = "© Captain"
                # Captain gets double, so their actual contribution = pts * 2
            elif eid == g["vice_captain"]:
                role = "VC"
            lines.append(f"| {i} | {pname} | {pos_str} | {pts} | {role} |")
        lines.append("")

        # Bench
        lines.append("**Bench**:")
        lines.append("")
        lines.append("| # | Player | Pos | Pts |")
        lines.append("|--:|:-------|:---:|----:|")
        for i, eid in enumerate(g["bench"], 1):
            pname = name_map.get(eid, str(eid))
            pos = loader.get_player_position(eid)
            pos_str = pos.name if pos else "?"
            pts = g["player_pts"].get(eid, 0)
            lines.append(f"| {i} | {pname} | {pos_str} | {pts} |")
        lines.append("")

        # Auto-subs
        if g["auto_subs"]:
            lines.append("**Auto-subs**:")
            for out_eid, in_eid in g["auto_subs"]:
                out_name = name_map.get(out_eid, str(out_eid))
                in_name = name_map.get(in_eid, str(in_eid))
                lines.append(f"- {out_name} → {in_name}")
            lines.append("")

        # Captain failover
        if g["captain_failover"]:
            lines.append(f"**Captain failover**: Vice-captain {name_map.get(g['vice_captain'], '?')} got the armband")
            lines.append("")

        # Points
        hit_str = f" - {g['hit_cost']} hits" if g['hit_cost'] else ""
        lines.append(f"**Points**: {g['gross_points']} gross{hit_str}"
                      f" = **{g['net_points']} net** | "
                      f"Captain bonus: {g['captain_points']} | "
                      f"Bench pts: {g['bench_points']}")
        lines.append(f"**Cumulative**: {g['cumulative']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Write report
    report_path = Path("reports/season_report_2024-25.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")
    print(f"Final score: {state.total_points} net points")


if __name__ == "__main__":
    main()
