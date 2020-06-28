import logging
import os

from serpapi.google_search_results import GoogleSearchResults

from ..utils import sync_to_async, registry

logger = logging.getLogger(__name__)


@registry.skill('google-search')
async def google_search(bot, user, query):
    params = dict(
        engine='google',
        q=query,
        api_key=os.getenv('SERPAPI_KEY'),
        gl='ru',
        hl='ru',
    )

    client = await sync_to_async(GoogleSearchResults, params)
    results = client.get_dict()
    answer = None
    logger.debug(f'serp results: {results}')
    if 'knowledge_graph' in results:
        knowledge_graph = results['knowledge_graph']
        if 'description' in knowledge_graph:
            answer = knowledge_graph['description']
            logger.debug(f'knowledge graph: {answer}')
    if answer is None and 'answer_box' in results:
        answer_box = results['answer_box']
        if 'result' in answer_box:
            answer = answer_box['result']
            logger.debug(f'answer box result: {answer}')
        elif 'snippet' in answer_box:
            answer = answer_box['snippet']
            logger.debug(f'answer box snippet: {answer}')
    if answer is None and results.get('organic_results'):
        answer = results['organic_results'][0].get('snippet')
    await bot.speak(answer or "что-то я не нашла")
