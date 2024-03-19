#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import logging
from typing import Sequence, Optional, List

from conf import agents_cfgs
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.agents.base_agent import AgentState, BaseAgent
from fairdiplomacy.pseudo_orders import PseudoOrders
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import Power, Timestamp, MessageDict
from fairdiplomacy.utils.typedefs import get_last_message
from parlai_diplomacy.utils.game2seq.format_helpers.misc import INF_SLEEP_TIME
from parlai_diplomacy.wrappers.factory import load_order_wrapper

from .parlai_message_handler import (
    ParlaiMessageHandler,
    ParlaiMessagePseudoOrdersCache,
    SleepSixTimesCache,
)
from .parlai_order_handler import ParlaiOrderHandler

import parse
from chatgpt.chatbot import Chatbot

sysyem_prompt = '''I want you to act as an expert player for Diplomacy game with press. Please help me with the game.
'''

info_input = '''I need you to play as the country {}. I will offer you the data about the game.
It contains information about current phase (key "CURRENT PHASE"), current units data (key "UNITS"), current centers data (key "CENTERS"), history orders for last phase in a dictionary (key "ORDER HISTORY"), the dialogue messages  for the last turn (key "LAST TURN DIALOGUE") and current turn(key "CURRENT TURN DIALOGUE"). The dialogue message is in the format of "sender -> recipient messages". 
The information string is:
{}
I will offer yor instructions futher and you will help me.
'''

message_input = '''
Please generate ONE message based on the information provided before in the following format. Do not write explanations.
YOUR POWER -> OTHER POWER: YOUR MESSAGES
'''

order_input = '''I need you to play as the country {}. I will offer you a string in JSON format now. It contains information about current phase (key "phase"), current units data (key "units"), current centers data (key "center"), history orders for last phase in a dictionary (key "order_history"). The string is in the format of "power: messages". The information string is:
{}
Please analysis the game based on the information and tell me what to do (in orders only) to win the game in the following format:
F ANK - CON
A SMY - ARM
A CON - BUL
Do not write explanations. You need to give me an order for each unit.
'''

order_input = '''Please analysis the game based on the information, and choose orders per location to excute in the following candidates:
{}
Respond me the orders you choose per line, remember you can only choose one order per location and do not explain.
'''

class ChatGPTAgentState(AgentState):
    def __init__(self):
        self.info_sent = False
        self.pseudo_orders_cache = ParlaiMessagePseudoOrdersCache()
        self.sleepsix_cache = SleepSixTimesCache()

class ChatGPTAgent(BaseAgent):
    def __init__(self, cfg, api_key=None):
        self.chatbot = Chatbot(api_key=api_key, system_prompt=sysyem_prompt, max_tokens=2000)
        self.message_handler = ParlaiMessageHandler(cfg.dialogue)
        self.model_orders = load_order_wrapper(cfg.model_orders)
        self.order_handler = ParlaiOrderHandler(self.model_orders)


    def initialize_state(self, power: Power) -> ChatGPTAgentState:
        return ChatGPTAgentState()

    def can_sleep(self) -> bool:
        return True

    def get_sleep_time(
        self, game: Game, power: Power, state: AgentState, recipient: Optional[Power] = None,
    ) -> Timestamp:
        # return Timestamp(2000)

        return self.messagec_handler.get_sleep_time(
            game, power, sleepsix_cache=state.sleepsix_cache, recipient=recipient,
        )

    def get_info(SELF, game: Game, power: Power):
        info = []

        phases = game.get_all_phases()

        current_phase = phases[-1]
        previous_phases = phases[:-1]

        info.append('CURRENT PHASE:\n{}'.format(current_phase.name))

        info.append('UNITS:\n')
        for key, val in current_phase.state['units'].items():
            info.append('{}: {};'.format(key, ', '.join(val)))

        info.append('CENTERS:\n')
        for key, val in current_phase.state['centers'].items():
            info.append('{}: {};'.format(key, ', '.join(val)))

        info.append('LAST TURN DIALOGUE:\n')
        if (len(previous_phases)):
            for msg in previous_phases[-1].messages.values():
                info.append('{} -> {}: {}\n'.format(msg['sender'], msg['recipient'], msg['message']))

        info.append('CURRENT TURN DIALOGUE:\n')
        for msg in current_phase.messages.values():
            info.append('{} -> {}: {}\n'.format(msg['sender'], msg['recipient'], msg['message']))

        info.append('ORDER HISTORY:\n')
        if(len(previous_phases)):
            for phase in previous_phases[-1:]:
                info.append('Phase {}'.format(phase.name))
                for key, val in phase.orders.items():
                    info.append('{}: {};'.format(key, ', '.join(val)))

        return '\n'.join(info)

    def get_info_json(SELF, game: Game, power: Power):
        info = {}
        phases = game.get_all_phases()

        current_phase = phases[-1]
        previous_phases = phases[:-1]

        info['phase'] = current_phase.name
        info['units'] = current_phase.state['units']
        info['centers'] = current_phase.state['centers']


        info['messages_received'] = []
        info['messages_sent'] = []

        if (len(previous_phases)):
            for msg in previous_phases[-1].messages.values():
                if (msg['sender'] == power):
                    info['messages_sent'].append('{}: {}'.format(msg['recipient'], msg['message']))
                elif (msg['sender'] == power):
                    info['messages_received'].append('{}: {}'.format(msg['sender'], msg['message']))

        info['order_history'] = {}
        if(len(previous_phases)):
            for phase in previous_phases[-1:]:
                info['order_history'][phase.name] = phase.orders
        return json.dumps(info)

    def parse_messages(self, game: Game, power_name: Power, timestamp: Timestamp, messages: str):
        #{"sender": "ITALY", "recipient": "AUSTRIA", "message": "Hey Austria! Are you down to run an AI? I really love playing that alliance as either of those two countries.", "phase": "S1901M", "time_sent": 2987}
        output_msg = {}
        for message in messages.split('\n'):
            try:
                recipient, msg = parse.parse('{}: {}', message).fixed
                # sender = sender.upper()
                recipient = recipient.upper()
                flag = False
                for power in POWERS:
                    if (power == power_name):
                        continue
                    if (power in recipient):
                        recipient = power
                        output_msg = {"sender": power_name, "recipient": recipient, "message": msg, "phase": game.current_short_phase, "time_sent": timestamp}
                        return output_msg
            except Exception as e:
                logging.warning('[INVALID MESSAGES]: {} -> {}'.format(power_name, message))
        return output_msg

    def get_possible_orders(self, game:Game, power_name:POWERS):
        possible_orders = game.get_all_possible_orders()
        valid_orders = {}
        for loc in game.get_orderable_locations()[power_name]:
            for order in possible_orders[loc]:
                if(loc in valid_orders):
                    valid_orders[loc].append(order)
                else:
                    valid_orders[loc] = [order]
        return valid_orders

    def parse_orders(self, valid_orders: dict, orders: str):
        output_orders = []
        for loc, val in valid_orders.items():
            for order in val:
                if(order in orders):
                    output_orders.append(order)
                    break
        return output_orders

        '''
        possible_orders = game.get_all_possible_orders()
        valid_orders = []
        for loc in game.get_orderable_locations()[power_name]:
            for order in possible_orders[loc]:
                if (order in orders):
                    valid_orders.append(order)
        '''

        '''
        
        for order in orders.split('\n'):
            flag = 0
            for loc in game.get_orderable_locations()[power_name]:
                if (order in possible_orders[loc]):
                    valid_orders.append(order)
                    flag = 1
                    continue
            if (flag == 0):
                logging.warning('[INVALID ORDER]: {}'.format(order))
        '''

        return tuple(valid_orders)

    def generate_message(
        self,
        game: Game,
        power: Power,
        timestamp: Optional[Timestamp],
        state: AgentState,
        recipient: Optional[Power] = None,
        pseudo_orders: Optional[PseudoOrders] = None,
    ) -> Optional[MessageDict]:
        """Return a complete MessageDict, or None on failure

        Implementations must choose a recipient within.

        Implement a default here which full-press agents should override.
        """
        # if(state.info_sent):
        #     cur_mss = []
        #     cur_mss.append('CURRENT TURN DIALOGUE:\n')
        #     for msg in game.get_all_phases()[-1].messages.values():
        #         cur_mss.append('{} -> {}: {}\n'.format(msg['sender'], msg['recipient'], msg['message']))
        #     Cur_mss = '\n'.join(cur_mss)
        #     message_in = Cur_mss + message_input
        #     # pass
        # else:
        #     info = self.get_info(game, power)
        #     state.info = info
        #     state.info_sent = True
        #     #self.chatbot.conversation['default'].append({'role': 'user', 'content': info_input.format(power, info)})
        #     bot_info = info_input.format(power, info)
        #     self.chatbot.info = bot_info
        #     bot_reply = self.chatbot.reply(bot_info)
        #     message_in = message_input

        info = self.get_info(game, power)
        state.info = infoc
        state.info_sent = True
        # self.chatbot.conversation['default'].append({'role': 'user', 'content': info_input.format(power, info)})
        bot_info = info_input.format(power, info)
        self.chatbot.info = bot_info
        bot_reply = self.chatbot.reply(bot_info)
        message_in = message_input

        bot_reply = self.chatbot.reply(message_in)
        messages = self.parse_messages(game, power, timestamp, bot_reply)
        return messages

    def get_orders(self, game: Game, power: Power, state: AgentState):
        state.info_sent = False
        # info = state.info
        # while(True):
        #     valid_orders = self.get_possible_orders(game, power)
        #     str_orders = ''
        #     for key, val in valid_orders.items():
        #         str_orders += '{}: {}\n'.format(key, ', '.join(val))
        #
        #     bot_reply = self.chatbot.reply(order_input.format(str_orders))
        #
        #     orders = self.parse_orders(valid_orders, bot_reply)
        #     if(len(orders)):
        #         break
        #     else:
        #         self.chatbot.rollback(1)
        info = self.get_info(game, power)
        state.info = info
        state.info_sent = True
        bot_info = info_input.format(power, info)
        self.chatbot.info = bot_info
        bot_reply = self.chatbot.reply(bot_info)
        valid_orders = self.get_possible_orders(game, power)
        str_orders = ''
        for key, val in valid_orders.items():
            str_orders += '{}: {}\n'.format(key, ', '.join(val))

        # bot_reply = self.chatbot.reply(order_input.format(str_orders))
        bot_reply = self.chatbot.reply(order_input.format(str_orders))

        orders = self.parse_orders(valid_orders, bot_reply)

        return orders