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
from sqlalchemy.schema import MetaData, Index
from sqlalchemy.sql import select, delete
from sqlalchemy.sql.functions import min


class HeapSQL:
    def __init__(self, delete_all=True):
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
        if delete_all:
            self.delete()
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
        row = self.conn.execute(select([self.heapq.c.id,
                                        min(self.heapq.c.cost).label("min_cost"),
                                        self.heapq.c.count,
                                        self.heapq.c.data])
                                .where(self.heapq.c.instance == instance)
                                .group_by(self.heapq.c.id,
                                          self.heapq.c.count,
                                          self.heapq.c.data)).fetchone()
        if row is not None:
            self.conn.execute(delete(self.heapq).where(self.heapq.c.id == row.id))
            return row.min_cost, row.count, pickle.loads(row.data)
        else:
            return None, None, None


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
