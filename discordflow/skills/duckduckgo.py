import logging

import aiohttp

from ..utils import language, registry, _, strip_accents

logger = logging.getLogger(__name__)


def get_lang():
    return dict(ru='ru-ru', en='en-us')[language.get()]


@registry.skill()
async def duckduckgo(bot, userstate, query):
    await bot.say(await search(query))


async def search(query):
    async with aiohttp.ClientSession() as sess:
        async with sess.get('https://api.duckduckgo.com/', params=dict(q=query, format='json', pretty=1, kl=get_lang())) as resp:
            data = await resp.json(content_type=None)
            for field in ['AbstractText', 'Answer', 'Definition']:
                if field in data:
                    result = strip_accents(data[field])
                    if result:
                        logger.debug(f"{query} -> {result}")
                        return result
            else:
                return _("Ничего не нашёл :-(")
