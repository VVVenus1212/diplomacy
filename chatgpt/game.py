import random
import time
import json
import diplomacy
from diplomacy import Message
from diplomacy.utils.export import to_saved_game_format
from chatgpt.chatbot import Chatbot
import parse
import sys
import logging

import fairdiplomacy
from fairdiplomacy_external.game_to_html import game_to_html
from mylog import MyLog

logger = MyLog(name='log.txt', log_level=logging.INFO)

# Creating a game
# Alternatively, a map_name can be specified as an argument. e.g. Game(map_name='pure')
api_key = ""
sysyem_prompt = '''
I want you to act as an expert player for Diplomacy game with press. You are the country AUSTRIA. There are three seasons each year (spring, fall and winter). And you can only make adjustments in winter. 
I will offer you current states of the game (including current phase, unit distribution, history orders, received messages). You will tell me what I should do (in orders) and who I should speak with (including messages). Respond conversationally. Do not write explanations.
An example of your reply should be like:
Orders:
AUSTRIA 
A BUD - SER
A VIE - TYR
F TRI - ADR
Messages:
AUSTRIA -> GERMANY: Hey Germany! Just reaching out to say I’m happy to keep Tyr/Boh as DMZs if you are. I always like to see Germany doing well as Austria, so if there’s ever any way I can help you out, please let me know!
AUSTRIA -> TURKEY: Hi Turkey! I think the AT is an extremely underrated alliance, especially since nobody ever expects it. Would you be down for going that route this game? I could try to get you Rumania or Sevastopol this year.
'''

'''
An example of the JSON string is like:
{'phase': 'F1901M', 'units': {'AUSTRIA': ['A VIE', 'F TRI', 'A RUM'], 'ENGLAND': ['F EDI', 'F LON', 'A YOR'], 'FRANCE': ['F BRE', 'A MAR', 'A PAR'], 'GERMANY': ['F HOL', 'A KIE', 'A SIL'], 'ITALY': ['F NAP', 'A APU', 'A ROM'], 'RUSSIA': ['A WAR', 'A MOS', 'F BLA', 'F LVN'], 'TURKEY': ['F ANK', 'A CON', 'A SYR']}, 'centers': {'AUSTRIA': ['BUD', 'TRI', 'VIE'], 'ENGLAND': ['EDI', 'LON', 'LVP'], 'FRANCE': ['BRE', 'MAR', 'PAR'], 'GERMANY': ['BER', 'KIE', 'MUN'], 'ITALY': ['NAP', 'ROM', 'VEN'], 'RUSSIA': ['MOS', 'SEV', 'STP', 'WAR'], 'TURKEY': ['ANK', 'CON', 'SMY']}, 'messages': [‘ENGLAND -> AUSTRIA: Can we ally?', ‘TURKEY -> AUSTRIA: Greetings!'], 'order_history': {'S1901M': {'AUSTRIA': ['A BUD - RUM', 'F TRI S A VEN', 'A VIE S A VEN - TRI'], 'ENGLAND': ['F EDI - NTH', 'F LON - NTH', 'A LVP - YOR'], 'FRANCE': ['F BRE H', 'A MAR S F BRE - GAS', 'A PAR S A MUN - BUR'], 'GERMANY': ['A BER - KIE', 'F KIE - HOL', 'A MUN - SIL'], 'ITALY': ['F NAP H', 'A ROM - APU', 'A VEN - ROM'], 'RUSSIA': ['A MOS S F SEV', 'F SEV - BLA', 'F STP/SC - LVN', 'A WAR S A BUD - GAL'], 'TURKEY': ['F ANK S A SMY - ARM', 'A CON S A SMY - ANK', 'A SMY - SYR']}}}
'''

sysyem_prompt = '''
I want you to act as an expert player for Diplomacy game with press. You are the country {}.
I will offer you a string in JSON format. It contains information about current phase (key "phase"), current units data (key "units"), current centers data (key "center"), history orders for last phase in a dictionary (key "order_history"), the historic sent messages (key "messages_sent" ) and received messages (key "messages_received") for the last phase. The string is in the format of "power: messages".
You will tell me what I should do (in orders) and who I should speak with (including messages). Respond conversationally. Do not write explanations.
'''

sysyem_prompt_austria = '''
I want you to act as an expert player for Diplomacy game with press. You are the country AUSTRIA.
I will offer you a string in JSON format. It contains information about current phase (key "phase"), current units data (key "units"), current centers data (key "center"), history orders for last phase in a dictionary (key "order_history"), the historic sent messages (key "messages_sent" ) and received messages (key "messages_received") for the last phase. The string is in the format of "power: messages".
You will tell me what I should do (in orders) and who I should speak with (including messages). Respond conversationally. Do not write explanations.
'''

sysyem_prompt_england = '''
I want you to act as an expert player for Diplomacy game with press. You are the country ITALY.
I will offer you a string in JSON format. It contains information about current phase (key "phase"), current units data (key "units"), current centers data (key "center"), history orders for last phase in a dictionary (key "order_history"), the historic sent messages (key "messages_sent" ) and received messages (key "messages_received") for the last phase. The string is in the format of "power: messages".
You will tell me what I should do (in orders) and who I should speak with (including messages). Respond conversationally. Do not write explanations.
'''

bot_input_prompt = '''
You should Analysis the current situation first, and send messages to other power. Think carefully what messages (to which power) you are going to send and what orders you should make to get more supply centers. 
Your reply should be in the following format:
Analysis:
Your analysis about current situation
Orders:
F EDI - NWG
A LVP - YOR
F LON S A LVP - YOR
Messages:
OTHER POWER: YOUR MESSAGES
OTHER POWER: YOUR MESSAGES
'''

def obtain_info(game, power_name):
    info = {}
    current_phase = diplomacy.Game.get_phase_data(game)
    previous_phases = diplomacy.Game.get_phase_history(game)

    info['phase'] = current_phase.name
    info['units'] = current_phase.state['units']
    info['centers'] = current_phase.state['centers']


    info['messages_received'] = []
    info['messages_sent'] = []

    if(len(previous_phases)):
        for msg in previous_phases[-1].messages.values():
            if(msg.sender == power_name):
                info['messages_sent'].append('{}: {}'.format(msg.recipient, msg.message))
            elif(msg.recipient == power_name):
                info['messages_received'].append('{}: {}'.format(msg.sender, msg.message))

            #info['messages'].append('{} -> {}: {}'.format(msg.sender, msg.recipient, msg.message))

    info['order_history'] = {}
    for phase in previous_phases[-2:]:
        info['order_history'][phase.name] = phase.orders



    return info

def parse_orders(game, power_name, orders):
    possible_orders = game.get_all_possible_orders()

    valid_orders = []
    for order in orders.split('\n'):
        flag = 0
        for loc in game.get_orderable_locations(power_name):
            if(order in possible_orders[loc]):
                valid_orders.append(order)
                flag = 1
                continue
        if(flag == 0):
            logger.warning('[INVALID ORDER]: {}'.format(order))

    return valid_orders


def parse_messages(game, power_name, messages):
    for message in messages.split('\n'):
        try:
            if(len(message)):
                #sender, recipient, msg = parse.parse('{} -> {}: {}', message).fixed
                recipient, msg = parse.parse('{}: {}', message).fixed
                #sender = sender.upper()
                recipient = recipient.upper()
                flag = False
                for power in game.powers.keys():
                    if(power == power_name):
                        continue
                    if(power in recipient):
                        recipient = power
                        flag = True
                        break
                if(flag):
                    game.add_message(Message(sender=power_name, recipient=recipient, message=msg, phase=game.get_current_phase(), time_sent=int(time.time())))
                    logger.info('[SEND MESSAGES]: {} -> {}'.format(power_name, message))
                    time.sleep(1)
                else:
                    logger.warning('[INVALID MESSAGES]: {} -> {}'.format(power_name, message))
        except Exception as e:
            logger.warning('[INVALID MESSAGES]: {} -> {}'.format(power_name, message))
    return messages

def main():
    #player = DipNetSLPlayer()
    game = diplomacy.Game()

    num_bots = 7
    bots = []
    bot_powers = []

    for i, power in enumerate(game.powers.keys()):
        bot_powers.append(power)
        bots.append(Chatbot(api_key=api_key, system_prompt=sysyem_prompt.format(power), max_tokens=2000, logger=logger))

    while not game.is_game_done:
        current_phase = diplomacy.Game.get_phase_data(game).name
        # orders = yield {power_name: player.get_orders(game, power_name) for power_name in game.powers}
        # Getting the list of possible orders for all locations
        possible_orders = game.get_all_possible_orders()

        phase_orders = {}

        # For each power, randomly sampling a valid order
        for power_name, power in game.powers.items():
            if(power_name in bot_powers):
                continue
            power_orders = [random.choice(possible_orders[loc]) for loc in game.get_orderable_locations(power_name)
                            if possible_orders[loc]]
            phase_orders[power_name] = power_orders

        logger.info("[CURRENT PHASE]: {}".format(current_phase))
        for i, bot in enumerate(bots):
            info_bot = obtain_info(game, bot_powers[i])

            bot_inputs = '{}\n{}'.format(json.dumps(info_bot), bot_input_prompt)

            #logger.info("[METADATA]: {}".format(json.dumps(info)))

            if(current_phase[0] == 'W'):
                bot_policy = [random.choice(possible_orders[loc]) for loc in game.get_orderable_locations(bot_powers[i]) if possible_orders[loc]]
                phase_orders[bot_powers[i]] = bot_policy
            else:
                while (True):
                    try:
                        bot_reply = bot.reply(bot_inputs)

                        bot_policy = parse_orders(game, bot_powers[i], bot_reply.split('Messages:')[0].split('Orders:')[1].strip())
                        bot_messages = parse_messages(game, bot_powers[i], bot_reply.split('Messages:')[1].strip())
                        phase_orders[bot_powers[i]] = bot_policy
                        break
                    except Exception as e:
                        time.sleep(60)


        # Processing the game to move to the next phase

        for power_name, power_orders in phase_orders.items():
            game.set_orders(power_name, power_orders)
            logger.info("[EXECUTE ORDER][{}]: {}".format(power_name, ', '.join(power_orders)))
        game.process()

        if(current_phase == 'S1906M' or int(current_phase[1:-1]) >= 1906):
            game.phase = 'COMPLETED'

    # Exporting the game to disk to visualize (game is appended to file)
    # Alternatively, we can do >> file.write(json.dumps(to_saved_game_format(game)))

    game_json = to_saved_game_format(game)
    game = fairdiplomacy.pydipcc.Game.from_json(json.dumps(game_json))
    html = game_to_html(game, title="ChatGPT(AUSTRIA) vs ChatGPT(ITALY) vs 5 Random Bots", annotations=None, filter1=None)
    with open('game_chat_chat.html', "w") as f:
        f.write(html)
if __name__ == '__main__':
    main()
