import logging
from pprint import pformat

from aidungeon.client import connect_to_aidungeon
from aidungeon.story.utils import player_won, player_died

from ..utils import registry, language
from ..google import translate

logger = logging.getLogger(__name__)


@registry.skill()
async def aidungeon(bot, state, initial: str = None):
    async with connect_to_aidungeon('http://localhost:8008', state.story) as aidungeon:
        bot_text = str(aidungeon.story)
        while True:
            state.story = aidungeon.story.to_dict()
            logger.debug(f"Story: {pformat(state.story)}")
            if player_won(bot_text):
                await bot.say(f"{bot_text}. Congratulations, you won.")
                return
            elif player_died(bot_text):
                await bot.say(f"{bot_text}. You lose, game over.")
                return
            if language.get() != 'en':
                bot_text = await translate('en', language.get(), bot_text)
            user_text = await bot.ask(bot_text, timeout=10)
            if user_text[:1] == '/' and len(user_text) > 1:
                bot_text = process_command(aidungeon, user_text[:1])
            else:
                if language.get() != 'en':
                    user_text = await translate(language.get(), 'en', user_text)
                bot_text = await aidungeon.process_action(user_text)


def process_command(aidungeon, command):
    if command == 'restart':
        return aidungeon.restart()
        return f"Game restarted.\n{aidungeon.story.story_start}"

    elif command == 'revert':
        return aidungeon.revert()

    else:
        return "Unknown command"
