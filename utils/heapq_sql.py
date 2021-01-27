#
#
# sites:
# https://www.pgadmin.org/docs/pgadmin4/latest/container_deployment.html
#
# docker run --name postgres -e POSTGRES_PASSWORD=aldebaran -d postgres
# docker pull dpage/pgadmin4
# docker run -p 80:80 \
#    -e 'PGADMIN_DEFAULT_EMAIL=user@domain.com' \
#    -e 'PGADMIN_DEFAULT_PASSWORD=aldebaran' \
#    -d dpage/pgadmin4

import uuid
import pickle
import base64
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, Float, Binary
from sqlalchemy.schema import MetaData, Index, DropTable
from sqlalchemy.sql import select


class HeapSQL:
    def __init__(self):
        self.name = 'priority_queue'
        self.cli = create_engine("postgresql://postgres:aldebaran@172.17.0.3/heapq", echo=False)
        self.conn = self.cli.connect()
        self.meta = MetaData()
        self.heapq = Table(
            self.name, self.meta,
            Column('id', Integer, primary_key=True),
            Column('instance', String, index=True),  # UUID
            Column('cost', Float, index=True),
            Column('count', Integer),
            Column('data', Binary),
            Index('idx_key', 'instance', 'cost')
        )
        self.meta.create_all(self.cli)
        # try:
        # except es.exceptions.RequestError as e:
        #     print(f'HeapSQL: create index: Runtime error: {e}')

    def delete(self):
        try:
            self.heapq.drop(self.cli)
        except Exception as e:
            print(f'HeapSQL: delete: Runtime error: {e}')

    def heappush(self, instance: str, cost: float, count: int, data):
        """Push item onto heap, maintaining the heap invariant."""
        # The Base64 encoded binary value must not have embedded newlines \n.
        item = {'instance': instance, 'cost': cost, 'count': count, 'data': pickle.dumps(data)}
        self.conn.execute(self.heapq.insert(values=item))

    def heappop(self, instance):
        """Pop the smallest item off the heap, maintaining the heap invariant."""
        row = self.conn.execute(select([self.heapq]).where(self.heapq.c.instance==instance)).fetchone()
        id = row[self.heapq.c.id]
        return row[self.heapq.c.cost], row[self.heapq.c.count], 0

        #s = Search(using=self.cli, index=self.name).sort('cost')
        #s.aggs.metric("min_cost", "min", field="cost")
        #response = s.execute()
        #return response.aggregations.min_cost.value, response.hits[0].count, \
        #       pickle.loads(base64.b64decode(response.hits[0].node.encode('UTF-8')), encoding='UTF-8')


if __name__ == "__main__":
    import random

    h = HeapSQL()
    #for i in range(100):
    #    for j in range(10):
    #        instance = uuid.uuid1().hex
    #        h.heappush(instance=instance, cost=random.uniform(0, 1), count=i, data='new node')
    #    print('heappush: ', self.conn.execute(, i)
    min_cost, count, node = h.heappop(instance='6f69128c602c11ebbe46309c2347a23c')
    print(min_cost, count, node)
    # h.delete()
