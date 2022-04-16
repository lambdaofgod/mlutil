# this code is adapted from
# https://github.com/searx/searx/blob/master/searx_extra/standalone_searx.py

import searx
import searx.preferences
import searx.query
import searx.search
import searx.webadapter
import attr
from typing import Optional, List, Dict, Any

EngineCategoriesVar = Optional[List[str]]

searx.search.initialize()


def get_searx_results(
    query,
    category: str = "general",
    pageno: str = 1,
    lang: str = "all",
    timerange: str = "any",
    safesearch: str = "0",
    only_results: bool = True,  # return only website results and not answers, infoboxes
    **kwargs
):

    engine_cs = list(searx.engines.categories.keys())
    search_results = _get_search_query(
        query,
        category,
        pageno,
        lang,
        timerange,
        safesearch,
        engine_categories=engine_cs,
    )
    res_dict = _to_dict(search_results)
    if only_results:
        return res_dict["results"]
    else:
        return res_dict


def get_searx_definition(query, **kwargs):
    results = get_searx_results(query, only_results=False, **kwargs)
    definition_results = get_searx_results(
        "define " + query, only_results=False, **kwargs
    )
    has_answer = len(results["answers"])
    wikipedia_results = get_wiki_results(results) + get_wiki_results(definition_results)
    has_wikipedia_definition = len(wikipedia_results) > 0
    if has_answer:
        answer = results["answers"][0]
        return answer
    elif has_wikipedia_definition:
        return wikipedia_results[0]["content"]
    else:
        return definition_results[0]["content"]


def get_wiki_results(results):
    return [res for res in results["results"] if "wikipedia" in res["url"]]


def _get_search_query(
    query: str,
    category: str = "general",
    pageno: str = 1,
    lang: str = "all",
    timerange: str = "week",
    safesearch: str = "0",
    engine_categories: EngineCategoriesVar = None,
) -> searx.search.SearchQuery:
    """Get  search results for the query"""
    if engine_categories is None:
        engine_categories = list(searx.engines.categories.keys())
    try:
        category = category.decode("utf-8")
    except AttributeError:
        category = category
    form = {
        "q": query,
        "categories": category,
        "pageno": str(pageno),
        "language": lang,
        "engines": "google brave duckduckgo",
    }
    if timerange:
        form["timerange"] = timerange
    preferences = searx.preferences.Preferences(
        ["oscar"], engine_categories, searx.engines.engines, []
    )
    preferences.key_value_settings["safesearch"].parse(safesearch)

    search_query = searx.webadapter.get_search_query_from_webapp(preferences, form)[0]
    return search_query


def _to_dict(search_query: searx.search.SearchQuery) -> Dict[str, Any]:
    """Get result from parsed arguments."""
    result_container = searx.search.Search(search_query).search()
    result_container_json = {
        "search": {
            "q": search_query.query,
            "pageno": search_query.pageno,
            "lang": search_query.lang,
            "safesearch": search_query.safesearch,
            "timerange": search_query.time_range,
        },
        "results": _no_parsed_url(result_container.get_ordered_results()),
        "infoboxes": result_container.infoboxes,
        "suggestions": list(result_container.suggestions),
        "answers": list(result_container.answers),
        "paging": result_container.paging,
        "results_number": result_container.results_number(),
    }
    return result_container_json


def _no_parsed_url(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove parsed url from dict."""
    for result in results:
        del result["parsed_url"]
    return results
