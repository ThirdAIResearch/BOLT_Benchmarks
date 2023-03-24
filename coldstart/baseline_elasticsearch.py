from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd
import time

# docker run --rm -p 9200:9200 -p 9300:9300 -e "xpack.security.enabled=false" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:8.3.3

# supervised data has a header of LABEL_IDS,QUERY with a comma delimiter and a
# ";" delimiter for the LABEL_IDS

# unsupervised data has a LABEL_IDS column and a DESCRIPTION column, otherwise 
# same format as above

es = Elasticsearch("http://localhost:9200", timeout=100)
print(es.info().body)


for dataset in ["wayfair"]:
    dataset_dir = f"{dataset}"
    index_name = dataset
    df = pd.read_csv(f"{dataset_dir}/reformatted_trn_unsupervised.csv")

    print(df.shape[0])

    mappings = {
        "properties": {
            "description": {"type": "text", "analyzer": "standard"},
        }
    }

    es.indices.delete(index=index_name, ignore=[400, 404])
    print(es.indices.create(index=index_name, mappings=mappings))

    bulk_data = []
    for i, row in df.iterrows():
        # for product_id in row["LABEL_IDS"].split(";"):
        #     if type(row["DESCRIPTION"]) == type(""):
        #         bulk_data.append(
        #             {
        #                 "_index": index_name,
        #                 "_id": int(product_id),
        #                 "_source": {
        #                     "description": row["DESCRIPTION"],
        #                 }
        #             }
        #         )
        if type(row["DESCRIPTION"]) == type(""):
            bulk_data.append(
                {
                    "_index": index_name,
                    "_id": int(row["LABEL_IDS"]),
                    "_source": {
                        "description": row["DESCRIPTION"],
                    }
                }
            )

    # print(len(bulk_data))

    bulk(es, bulk_data)

    es.indices.refresh(index=index_name)
    # print(es.cat.count(index=index_name, format="json"))

    df = pd.read_csv(f"{dataset_dir}/reformatted_tst_supervised.csv")

    precision_at_1 = 0
    recall = 0
    total_time = 0
    for i, row in df.iterrows():
        start = time.time()
        resp = es.search(
            index=index_name,
            body={
                "size": 100,
                "query": {
                    "match": {
                    "description": {
                        "query": row["QUERY"]
                    }
                    }
                }
            }
        )
        end = time.time()
        total_time += end - start

        relevant_products = set([product_id for product_id in str(row["LABEL_IDS"]).split(";")])

        # products are sorted by elastic search score
        recommended_products = [hit['_id'] for hit in resp['hits']['hits']]
        
        if len(recommended_products) > 0 and recommended_products[0] in relevant_products:
            precision_at_1 += 1

        partial_recall = 0
        for recommended_product in recommended_products:
            if recommended_product in relevant_products:
                partial_recall += 1
        partial_recall /= len(relevant_products)
        recall += partial_recall

    print(f"Recall@100 for dataset {dataset} ", recall / df.shape[0])
    print(f"Precision@1 for dataset {dataset} ", precision_at_1 / df.shape[0])
    print(total_time / df.shape[0])
