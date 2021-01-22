#
# priority queues on elastic search apache curator
# sites:
# https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html
# https://curator.readthedocs.io/en/latest/index.html
#
import elasticsearch as es
import curator


class HeapElastic:
    def __init__(self):
        self.name = 'priority_queue'
        self.cli = es.Elasticsearch(host='localhost', port=9200)
        self.co = curator.CreateIndex(cli, self.name, extra_settings={
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "items": {
                    "properties": {
                        "cost": {"type": "float"},
                        "count": {"type": "integer"},
                        "node": {"type": "binary"},
                    }}}})

    def __del__(self):
        return 0

    def heappush(self, item):
        """Push item onto heap, maintaining the heap invariant."""
        res = cli.index(index=self.name, id=1, body={'key': 'antonio'})

    def heappop(self):
        """Pop the smallest item off the heap, maintaining the heap invariant."""
        res = cli.get(index=self.name, id=1)
        return res
