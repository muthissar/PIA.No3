import json
import os
from pymongo import MongoClient
from sshtunnel import SSHTunnelForwarder


class LsdbMongo:
    def __init__(self):
        self.credentials = self.load_credentials()

        self.server = SSHTunnelForwarder(
            self.credentials['SERVER_ADDRESS'],
            ssh_username=self.credentials['SSH_USER_NAME'],
            remote_bind_address=('127.0.0.1', 27017),
            ssh_pkey=self.credentials['SSH_PKEY']
        )
        self.server.start()

        self.client = MongoClient('localhost',
                                  self.server.local_bind_port,
                                  )

    def get_db(self):
        # Cannot connect directly to lsdb and then performing auth
        db = self.client.get_database('admin')
        db.authenticate(self.credentials['LOGIN_READONLY'],
                        self.credentials['PASSWORD_READONLY'],
                        mechanism='SCRAM-SHA-1')
        lsdb = self.client.get_database('lsdb')
        return lsdb

    def get_songbook_leadsheets_cursor(self, db):
        """Return a cursor all songbook leadsheets, excluding user input ones
        """
        return db.leadsheets.find(
            {'source': {"$ne": "51b6fe4067ca227d25665b0e"}})

    def close(self):
        self.client.close()
        self.server.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def load_credentials():
        if not os.path.exists('passwords.json'):
            empty_credentials = {
                "SERVER_ADDRESS": "",
                "SSH_USER_NAME": "",
                "SSH_PKEY": "",
                "LOGIN_READONLY": "",
                "PASSWORD_READONLY": ""
            }
            with open('passwords.json', 'w') as f:
                json.dump(empty_credentials, f, indent=2)
                print('An empty passwords.json file has been generated in:')
                print(os.path.dirname(os.path.abspath(__file__)))
                print('Please edit this file.')

        credentials = json.load(open('passwords.json', 'r'))
        return credentials


if __name__ == '__main__':
    lsdb_client = LsdbMongo()
    db = lsdb_client.get_db()
    # cursor = lsdb_client.get_songbook_leadsheets_cursor(db)
    # print(next(cursor))
    lsdb_client.close()
