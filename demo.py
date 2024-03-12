import heyhi
import logging
import socket
import os
from run import TASKS
import argparse
import pathlib
from heyhi import util,checkpoint_repo,conf,setup_logging
from heyhi.run import get_exp_dir
import functools
import torch
import numpy as np

from fairdiplomacy.agents import build_agent_from_cfg
from fairdiplomacy.agents.chatgpt_agent import ChatGPTAgent
from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.compare_agents import run_1v6_trial, run_1v6_trial_multiprocess
from fairdiplomacy.compare_agent_population import run_population_trial
from fairdiplomacy.models.base_strategy_model import train_sl
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.utils.yearprob import parse_year_spring_prob_of_ending
from fairdiplomacy import pydipcc
from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from fairdiplomacy.env import Env, OneSixPolicyProfile, SharedPolicyProfile
from fairdiplomacy.viz.meta_annotations.api import maybe_kickoff_annotations
from fairdiplomacy_external.game_to_html import game_to_html
from fairdiplomacy.situation_check import run_situation_check_from_cfg
from fairdiplomacy.typedefs import Power

from fairdiplomacy.agents.player import Player
from animation import html_to_animation

api_key = "YOUR_API_KEY"

def arg(override):
    # heyhi.parse_args_and_maybe_launch(main)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--cfg",  type=pathlib.Path,
                        default=pathlib.Path("conf/c01_ag_cmp/cmp.prototxt"))
    parser.add_argument("--adhoc", action="store_true", default=True)
    parser.add_argument(
        "--force", action="store_true", help="Do not ask confirmation in restart mode"
    )
    parser.add_argument(
        "--mode", choices=util.MODES, help="See heyhi/util.py for mode definitions."
    )
    checkpoint_repo.add_parser_arg(parser)
    parser.add_argument(
        "--exp_id_pattern_override",
        default=util.EXP_ID_PATTERN,
        help="A pattern to construct exp_id. Job's data is stored in <exp_root>/<exp_id>",
    )
    parser.add_argument("--out", help="Alias for exp_id_pattern_override. Overrides it if set.")
    parser.add_argument(
        "--print", help="If set, will print composed config and exit", action="store_true"
    )
    parser.add_argument(
        "--print-flat",
        help="If set, will print composed config as a list of redefines and exit. This command lists only explicitly defined flags",
        action="store_true",
    )
    parser.add_argument(
        "--print-flat-all",
        help="If set, will print composed config as a list of redefines and exit. This command lists all possible flags",
        action="store_true",
    )
    parser.add_argument("--log-level", default="INFO", choices=["ERROR", "WARN", "INFO", "DEBUG"])
    args, overrides = parser.parse_known_args()

    if args.out:
        args.exp_id_pattern_override = args.out
    del args.out
    if args.mode is None:
        args.mode = "restart" if args.adhoc else "gentle_start"

    # overrides = [x.lstrip("-") for x in overrides]
    # overrides = ['Iagent_one=agents/ablations/cicero_imitation_only.prototxt', 'Iagent_six=agents/ablations/cicero_imitation_only.prototxt',
    #              'power_one=TURKEY']
    overrides = override

    log_level = "INFO"
    util.setup_logging(console_level=log_level)
    logging.info("Config: %s", args.cfg)
    logging.info("Overrides: %s", overrides)

    exp_root = get_exp_dir("diplomacy")

    exp_id = util.get_exp_id(args.cfg, overrides, args.adhoc, exp_id_pattern=args.exp_id_pattern_override)
    exp_handle = util.ExperimentDir(exp_root / exp_id, exp_id=exp_id)
    need_run = util.handle_dst(exp_handle, args.mode, force=args.force)
    logging.info("Exp dir: %s", exp_handle.exp_path)
    logging.info("Job status [before run]: %s", exp_handle.get_status())

    if need_run:
        # Only checkpoint if we actually need a new run.
        # Specially disable checkpointing by default in the case of adhoc runs, will checkpoint
        # if the user explicitly specifies a non default checkpoint path
        if args.adhoc and args.checkpoint == checkpoint_repo.DEFAULT_CHECKPOINT:
            ckpt_dir = ""
        else:
            ckpt_dir = checkpoint_repo.handle_parser_arg(args.checkpoint, exp_handle.exp_path)
        # util.run_with_config(main, exp_handle, cfg, overrides, ckpt_dir, log_level)
        setup_logging(console_level=log_level)
        task, meta_cfg = conf.load_root_proto_message(args.cfg, overrides)
        cfg = getattr(meta_cfg, task)
        exp_handle.exp_path.mkdir(exist_ok=True, parents=True)
        old_cwd = os.getcwd()
        logging.info(f"Changing cwd to {exp_handle.exp_path}")
        os.chdir(exp_handle.exp_path)
        os.symlink(pathlib.Path(old_cwd) / "models", exp_handle.exp_path / "models")

        if hasattr(cfg, "launcher"):
            if not cfg.launcher.WhichOneof("launcher"):
                cfg.launcher.local.use_local = True
            launcher_type = cfg.launcher.WhichOneof("launcher")
            launcher_cfg = getattr(cfg.launcher, launcher_type)
        else:
            launcher_type = "local"
            launcher_cfg = None
        assert launcher_type in ("local", "slurm"), launcher_type

        conf.save_config(meta_cfg, pathlib.Path("config_meta.prototxt"))
        conf.save_config(cfg, pathlib.Path("config.prototxt"))

    logging.info(f"Machine IP Address: {socket.gethostbyname(socket.gethostname())}")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logging.info(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

    if not hasattr(cfg, "heyhi_patched"):
        raise RuntimeError("Run `make protos`")
    cfg = cfg.to_frozen()
    heyhi.setup_logging(console_level=log_level)
    logging.info("Cwd: %s", os.getcwd())
    logging.info("Task: %s", task)
    logging.info("Cfg:\n%s", cfg)
    logging.debug("Cfg (with defaults):\n%s", cfg.to_str_with_defaults())
    heyhi.log_git_status()
    logging.info("Is on slurm: %s", heyhi.is_on_slurm())
    # logging.info("Job env: %s", heyhi.get_job_env())
    if heyhi.is_on_slurm():
        logging.info("Slurm job id: %s", heyhi.get_slurm_job_id())
    logging.info("Is master: %s", heyhi.is_master())
    if getattr(cfg, "use_default_requeue", False):
        heyhi.maybe_init_requeue_handler()

    return cfg

if __name__ == '__main__':
    overrides = ['Iagent_one=agents/ablations/cicero_imitation_only.prototxt',
                 'Iagent_six=agents/ablations/cicero_imitation_only.prototxt',
                 'power_one=TURKEY']

    cfg = arg(overrides) #参数
    # cfg.max_year = 1908
    #随机
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # with open("/data/webw6/ChatDiplomacy/data/cicero_redacted_games/game_433920_RUSSIA_EI.json", "r") as f:
    #     s = f.read()  # string
    #
    # game = pydipcc.Game.from_json(s)
    #
    # html = game_to_html(game, title="Example", annotations=None, filter1=None)

    #构建智能体
    #agent_one = build_agent_from_cfg(cfg.agent_one) # BQRE1PAgent
    agent_one = ChatGPTAgent(cfg=cfg.agent_one.agent,api_key=api_key)
    agent_six = build_agent_from_cfg(cfg.agent_six) # Par# laiFullPressAgent
    # agent_six = ChatGPTAgent(api_key=api_key)
    cf_agent = None
    power_string = cfg.power_one #代表势力

    game_obj = pydipcc.Game() #定义game
    if cfg.draw_on_stalemate_years is not None and cfg.draw_on_stalemate_years > 0:
        game_obj.set_draw_on_stalemate_years(cfg.draw_on_stalemate_years)

    #这里将7个智能体封装了
    policy_profile = OneSixPolicyProfile(
        agent_one=agent_one,
        agent_six=agent_six,
        agent_one_power=power_string,
        game=game_obj,
        share_strategy=cfg.share_strategy,
    )
    #一些参数
    variance_reduction_model = None
    if cfg.variance_reduction_model_path:
        variance_reduction_model = BaseStrategyModelWrapper(cfg.variance_reduction_model_path)
    year_spring_prob_of_ending = parse_year_spring_prob_of_ending(cfg.year_spring_prob_of_ending)

    #游戏运行环境
    env = Env(
        policy_profile=policy_profile,
        seed=cfg.seed,
        cf_agent=cf_agent,
        max_year=1910,
        max_msg_iters=cfg.max_msg_iters,
        game=game_obj,
        capture_logs=cfg.capture_logs,
        time_per_phase=10000,  #这个是限制一轮自由对话是多长时间，单位是厘秒。
        variance_reduction_model=variance_reduction_model,
        stop_when_power_is_dead=power_string
        if (cfg.stop_on_death and not cfg.use_shared_agent)
        else None,
        year_spring_prob_of_ending=year_spring_prob_of_ending,
    )

    #游戏循环
    max_turns = None
    env.turn_id = 0
    while not env.game.is_game_done:
        if max_turns and env.turn_id >= max_turns:
            break
        if (
                env.stop_when_power_is_dead is not None
                and env.stop_when_power_is_dead not in env.game.get_alive_powers()
        ):
            logging.info("Early stopping as agent %s is dead", env.stop_when_power_is_dead)
            break
        _, year, _ = env.game.phase.split()
        if int(year) > env.max_year:
            logging.info("Early stopping at %s due to reaching max year", year)
            break

        env.process_turn()  #内存封装很多层，主要是两部分：7个国家对话阶段和做出决策阶段
        env.turn_id += 1

    html = game_to_html(env.game, title="ChatGPT(TURKEY) vs 6 Cicero", annotations=None, filter1=None)
    animation_html = html_to_animation(html)

    with open('/root/wxj_test/diplomacy_cicero/game_0311.html', "w") as f:
        f.write(html)

    with open('/root/wxj_test/diplomacy_cicero/animation_game_0311.html', "w") as f:
        f.write(animation_html)

    print('a')










