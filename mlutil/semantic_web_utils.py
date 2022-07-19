from typing import List
import pandas as pd
import tqdm


def get_sparql_results(query):
    import SPARQLWrapper

    sparql = SPARQLWrapper.SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(SPARQLWrapper.JSON)
    return sparql.query().convert()


def get_related_concepts_with_abstracts_query(entity: str):
    return f"""
    SELECT DISTINCT ?child ?label ?abstract (group_concat(?child_type; separator=" ") AS ?child_type)
    WHERE {{
        {{
            {_make_relation_to_selected_subject_where_phrase("skos:broader{1,2}", entity, False)}
            UNION {_make_relation_to_selected_subject_where_phrase("dct:subject{1,2}|dbo:wikiPageWikiLink", entity, True)}
        }}
    }} GROUP BY ?child ?label ?abstract
    """


def _make_relation_to_selected_subject_where_phrase(
    relation, subject, query_abstract=True, filter_english_abstracts=True
):
    abstract_lang_filter = (
        " .\n        FILTER (lang(?abstract) = 'en')"
        if filter_english_abstracts
        else ""
    )
    abstract_query_part = (
        f".\n        ?child dbo:abstract ?abstract {abstract_lang_filter}"
        if query_abstract
        else ""
    )
    return f"""{{
        ?child {relation} {subject} .
        FILTER (lang(?label) = 'en') .
        ?child rdfs:label ?label {abstract_query_part} .
        ?child rdf:type ?child_type .
    }}"""


def get_related_concepts_results(concepts: List[str]):
    return sum(
        (
            get_sparql_results(get_related_concepts_with_abstracts_query(concept))[
                "results"
            ]["bindings"]
            for concept in tqdm.auto.tqdm(concepts)
        ),
        start=[],
    )


def get_results_with_abstract_field(results: List[dict]):
    return [res for res in results if res.get("abstract")]


def make_dataframe_from_results(results: List[dict]):
    return pd.DataFrame.from_records(
        [
            {key: rec[key]["value"] for key in rec.keys()}
            for rec in get_results_with_abstract_field(results)
        ]
    )


def filter_out_people(records_df, type_col="child_type"):
    return records_df[~records_df["child_type"].str.lower().str.contains("person")]
