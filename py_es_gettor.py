# 遍历ES的某个索引，获取全部文档
def es_iterate_all_documents(es, index, pagesize=1000, scroll_timeout="1m", **kwargs):
    """
    Helper to iterate ALL values from a single index
    Yields all the documents.
    """
    is_first = True
    while True:
        # Scroll next
        if is_first: # Initialize scroll
            result = es.search(index=index, scroll="1m", **kwargs, body={
                "size": pagesize
            })
            is_first = False
        else:
            result = es.scroll(body={
                "scroll_id": scroll_id,
                "scroll": scroll_timeout
            })
        scroll_id = result["_scroll_id"]
        hits = result["hits"]["hits"]
        # Stop after no more docs
        if not hits:
            break
        # Yield each entry
        yield from (hit['_source'] for hit in hits)

for item in es_iterate_all_documents(es, 'my_index'):
    print(item)

