#
# priority queues on elastic search apache curator
# sites:
# https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html
# https://elasticsearch-py.readthedocs.io/en/7.10.0/index.html
# https://elasticsearch-dsl.readthedocs.io
# https://coralogix.com/log-analytics-blog/42-elasticsearch-query-examples-hands-on-tutorial/
#
# TROPPO LENTO L'INSERIMENTO RECORDS
#
import elasticsearch as es
from elasticsearch_dsl import Search
import pickle
import base64


class HeapElastic:
    def __init__(self):
        self.name = 'priority_queue'
        self.cli = es.Elasticsearch(host='localhost', port=9200)
        self.ind = es.client.IndicesClient(self.cli)
        self.ret = None
        try:
            self.ret = self.ind.create(self.name, body={
                "settings": {
                    "number_of_shards": 4,
                    "number_of_replicas": 0
                },
                "mappings": {
                    "properties": {
                        "cost": {"type": "double"},
                        "count": {"type": "integer"},
                        "node": {"type": "binary"},
                    }}})
        except es.exceptions.RequestError as e:
            print(f'HeapElastic: create index: Runtime error: {e}')

    def delete(self):
        try:
            return self.ind.delete(self.name)
        except es.exceptions.RequestError as e:
            print(f'HeapElastic: create delete: Runtime error: {e}')

    def heappush(self, cost: float, count: int, data):
        """Push item onto heap, maintaining the heap invariant."""
        # The Base64 encoded binary value must not have embedded newlines \n.
        item = {'cost': cost, 'count': count, 'node': base64.b64encode(pickle.dumps(data)).decode('UTF-8')}
        return self.cli.index(self.name, body=item)

    def heappop(self):
        """Pop the smallest item off the heap, maintaining the heap invariant."""
        s = Search(using=self.cli, index=self.name).sort('cost')
        s.aggs.metric("min_cost", "min", field="cost")
        response = s.execute()
        return response.aggregations.min_cost.value, response.hits[0].count, \
               pickle.loads(base64.b64decode(response.hits[0].node.encode('UTF-8')), encoding='UTF-8')


if __name__ == "__main__":
    import random

    h = HeapElastic()
    # print(h.ret)
    for i in range(1000):
        res = h.heappush(cost=random.uniform(0, 1), count=i, data='new node')
        print('heappush: ', i)
    min_cost, count, node = h.heappop()
    print(min_cost, count, node)
    # print('Delete: ', h.delete())
