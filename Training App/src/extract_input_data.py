import os.path
import logging
logging.basicConfig(level=logging.INFO)
from sqlalchemy.engine import Connection
from src.sql_utils import (
    get_db_connection,
    _actual_create_table,
    get_table_summary,
)


def read_sql_file(name: str) -> str:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    dir_sql = os.path.join(this_dir, "sql")
    if not name.endswith(".sql"):
        name += ".sql"
    fn_sql = os.path.join(dir_sql, name)
    assert os.path.exists(fn_sql), f"File {fn_sql} does not exist"
    with open(fn_sql, "r") as f:
        return f.read()


def create_table_scripts_test(
    cnx: Connection,
    schema_name: str,
    do_the_job: bool = True,
    drop_if_exists: bool = False,
) -> str:
    """Test the scripts"""
    table_name = "scripts_test"
    if do_the_job:
        _actual_create_table(
            cnx,
            schema=schema_name,
            table_name=table_name,
            sql=read_sql_file(table_name),
            drop_if_exists=drop_if_exists,
        )
    return table_name


def main(
    *,
    fn_connection: str = None,
    schema_interim: str = "dev_yehuda",
    create_scripts_test: bool = True,
    create_all: bool = True,
    force_creation_if_exists: bool = True,
):
    """
    Generate feature tables.

    This script is used to generate the feature tables for the future modelling. All the tables are generated in the same
    schema. The default schema is `dev_boris`, but you can specify your own by using the `schema_interim` parameter.

    :param fn_connection: where the connection secrets are stored. The default is `../data/secrets/db_connect.json`.
     This is a JSON file with the following keys: "host", "port", "user", "password", "dbname"
    :param schema_interim: the target schema where we save the interim data
    :param create_scripts_test: create test table
    :param create_all: should we create all the tables. If this parameter is True, then all the other parameters are
    :param force_creation_if_exists: should we force the creation of the tables if they already exist.
    """

    if create_all:
        create_scripts_test = True

    arguments = dict(locals())
    create_summary_table = any([arguments[k] for k in arguments.keys() if k.startswith("create_")])
    create_last_available_data_table = create_summary_table

    if fn_connection is None:
        this_dir = os.path.abspath(os.path.split(__file__)[0])
        dir_data = os.path.abspath(os.path.join(this_dir, "../data"))
        assert os.path.exists(dir_data)
        dir_secrets = os.path.join(dir_data, "secrets")
        fn_connection = os.path.join(dir_secrets, "db_connect.json")
    cnx = get_db_connection(fn_connection)

    tables = [
        create_table_scripts_test(
            cnx,
            schema_interim,
            do_the_job=create_scripts_test,
            drop_if_exists=force_creation_if_exists,
        )
    ]

    print("Created tables:")
    for t in tables:
        table_summary = get_table_summary(cnx, schema_interim, t)
        print(table_summary)
        print("\n\n")


if __name__ == "__main__":
    main()
