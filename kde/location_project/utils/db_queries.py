import psycopg2
import ConfigParser

print 'DB_QUERIES: Loading database parameters'

## Loading the parameter
config = ConfigParser.RawConfigParser()
config.read('properties.conf')

USER = config.get('data_base', 'user')
HOST = config.get('data_base', 'host')
PASS = config.get('data_base', 'password')
DB_NAME = config.get('data_base', 'dbname')


def get_all_raw_data_for_user(uid,loc):
    """
    Returns the raw data for a user and location combo. This prevents us from getting user data for users who travel
    between areas.

     OUTPUT:
    --------
        1. table: List of tuples [(tweet_id, time_stamp, longitude, latitude), .... ]
    """
    conn = psycopg2.connect("host=%s dbname=%s user=%s password=%s" % (HOST, DB_NAME, USER, PASS))
    cur = conn.cursor()

    query = "SELECT tweet_id, timestamp, long, lat FROM twitter_data WHERE user_id = '%s' AND loc = '%s';" % (uid,loc)
    cur.execute(query)

    table = cur.fetchall()
    try:
        conn.close()
    except Exception:
        pass

    return table


def get_unique_user_ids_from_raw_data(loc):
    """
    Returns a list of user ids from the raw 'data' table

    INPUT:
    -------
        1. loc: location to get unique user ids from

     OUTPUT:
    --------
        1. uids: List of unique user ids
    """
    conn = psycopg2.connect("host=%s dbname=%s user=%s password=%s" % (HOST, DB_NAME, USER, PASS))
    cur = conn.cursor()

    query = "SELECT DISTINCT ON (user_id) user_id FROM twitter_data where loc = '%s';" % loc
    cur.execute(query)

    table = cur.fetchall()
    try:
        conn.close()
    except Exception:
        pass

    return [uid[0] for uid in table]


def insert_to_table(table, columns, values):
    """
    Inserts the values into table according to column. The values are string so it it doesn't matter
    to this function if it's one set of values or multiple, the caller is responsible for that.
    The caller is also responsible for the paranthesis in the right places in both columns and values.

     INPUT:
    -------
        1.table:
        2.columns:
        3.values:
    """
    conn = psycopg2.connect("host=%s  dbname=%s user=%s password=%s" % (HOST, DB_NAME, USER, PASS))
    cur = conn.cursor()
    insert_query = "INSERT INTO %s %s VALUES %s;" % (table, columns, values)
    cur.execute(insert_query)
    conn.commit()

    try:
        conn.close()
    except:
        pass
