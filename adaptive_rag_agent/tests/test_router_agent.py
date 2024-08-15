import unittest
from router_agent import routerAgent

class TestRouterAgent(unittest.TestCase):
    def test_route(self):
        router = routerAgent()
        self.assertEqual(router.route_question("Who will the Bears draft first in the NFL draft?"), {"datasource": "web_search"})
        self.assertEqual(router.route_question("What are the key components of an agent?"), {"datasource": "vectorstore"})

if __name__ == '__main__':
    unittest.main()