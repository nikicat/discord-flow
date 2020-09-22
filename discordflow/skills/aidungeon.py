import logging
from pprint import pformat

from aidungeon.client import connect_to_aidungeon

from ..utils import registry, language
from ..google import translate

logger = logging.getLogger(__name__)


@registry.skill()
async def aidungeon(bot, userstate, initial: str = None):
    async with connect_to_aidungeon('http://localhost:8008') as aidungeon:
        bot_text = str(aidungeon.story)
        while True:
            logger.debug(f"Story: {pformat(aidungeon.story.to_dict())}")
            if language.get() != 'en':
                bot_text = await translate('en', language.get(), bot_text)
            user_text = await bot.ask(bot_text, timeout=10)
            if language.get() != 'en':
                user_text = await translate(language.get(), 'en', user_text)
            bot_text = await aidungeon.ask(user_text)
