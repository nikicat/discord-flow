import logging
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from itertools import groupby

from ..utils import registry, EmptyUtterance

logger = logging.getLogger(__name__)


def get_next_letter(city):
    return re.sub('[ыьъ]', '', city)[-1]


def normalize_city(city):
    return city.lower().replace('-', '').replace(' ', '')


def load_cities():
    return {
        first_letter: list(cities)
        for first_letter, cities in groupby(
            sorted([normalize_city(line.split(';')[-1][1:-1]) for line in open('city_.csv').read().split('\n')[1:]]),
            key=lambda city: city and city[0],
        )
    }


@dataclass
class UserState:
    wins: int = 0
    loses: int = 0


class Lose(Exception):
    pass


@registry.skill('cities')
async def cities_skill(bot, user_state: UserState):
    user_state = user_state or UserState()
    cities = load_cities()
    orig_cities = deepcopy(cities)
    seen_cities = set()
    question = f"Давай! Твой счёт {user_state.wins}-{user_state.loses}. Называй город!"
    bot_city = None
    try:
        while True:
            try:
                user_city = normalize_city(await bot.ask(question, timeout=5, tries=3))
            except EmptyUtterance:
                raise Lose("Время вышло")
            if user_city == "повтори":
                continue
            user_letter = user_city[0]
            if user_city not in orig_cities[user_letter]:
                raise Lose("Такого города нет")
            elif bot_city and user_letter != (expected_user_letter := get_next_letter(bot_city)):
                raise Lose(f"Неправильно, ты должен был назвать город на букву {expected_user_letter}")
            elif user_city in seen_cities:
                raise Lose("Этот город уже называли")
            else:
                seen_cities.add(user_city)
                cities[user_letter].remove(user_city)
                bot_letter = get_next_letter(user_city)
                try:
                    bot_city = random.choice(cities[bot_letter])
                except IndexError:
                    user_state.wins += 1
                    await bot.say("Сдаюсь, ты выиграл в {user.wins} раз!")
                    break
                seen_cities.add(bot_city)
                cities[bot_letter].remove(bot_city)
                logger.debug(f"Bot city: {bot_city}")
                question = bot_city
    except Lose as exc:
        user_state.loses += 1
        await bot.say(f"{exc}. Ты проиграл в {user_state.loses}-й раз, ха-ха-ха!")
    return user_state
