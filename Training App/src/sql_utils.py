"""General SQL-related functions"""
import json
import os
from typing import Union, Iterable
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connection


def get_db_connection(fn_connection: str) -> Connection:
    """
    Get a connection to the database.

    This function assumes RedShift databases and uses postgresql_psycopg2 driver.

    :param fn_connection: A JSON file with the connection parameters
            (required keys: "host", "port", "dbname", "user", "password")
    :return: the connection object
    """

    assert os.path.exists(fn_connection)
    sql_config = json.load(open(fn_connection))
    host, port, dbname, user, password = [
        sql_config[k] for k in ["host", "port", "dbname", "user", "password"]
    ]
    cnx = sa.create_engine(
        sa.engine.url.URL.create(
            drivername="postgresql+psycopg2",
            username=user,
            password=password,
            host=host,
            port=port,
            database=dbname,
        )
    ).connect()
    return cnx


def generate_where_part(vals: Iterable, col_name: str) -> str:
    """For when you have a list of values and want to generate a WHERE part of an SQL query"""
    vals = [str(v) for v in vals]
    if not vals:
        # Empty value list, should not return anything
        ret = "FALSE"
    else:
        ret = ",".join(vals)
        ret = f"{col_name} IN ({ret})"
    return ret


def read_table(
    table_name: str, cnx: Connection, where: Union[None, str] = None, limit: int = None
) -> pd.DataFrame:
    """Read a table by its name, optionally with a WHERE clause and a LIMIT"""
    query = f"SELECT * FROM {table_name}"
    if where:
        query += f" WHERE {where}"
    if limit:
        query += f" LIMIT {limit}"
    ret = pd.read_sql(query, cnx)
    return ret


def get_tables_and_views(conx: Connection, schema: str = None) -> pd.DataFrame:
    """Get a dataframe of tables and views in a schema"""
    sql = f"""
        SELECT t.table_name, t.table_type
        FROM information_schema.tables t
    """
    if schema:
        sql += f" WHERE table_schema = '{schema}';"
    df_tables_and_views = pd.read_sql(sql, conx)
    return df_tables_and_views


def drop_table_if_exists(cnx: Connection, schema: str, table_name: str):
    sql = f"DROP TABLE IF EXISTS {schema}.{table_name}"
    cnx.execute(sql)


def sanity_check_df(df: pd.DataFrame) -> pd.DataFrame:
    """Perform several diagnostic heuristics on a dataframe. Return the results as a dataframe"""
    numeric_cols = set(df.select_dtypes(include=np.number).columns)
    cols = df.columns
    res = dict()
    for col in cols:
        vals = df[col]
        curr = dict()
        curr["n"] = f"{len(vals):8,d}"
        curr["n_unique"] = f"{vals.nunique(dropna=False):8,d}"
        curr["cardinality_fraction"] = np.round(vals.nunique() / len(vals), 3)
        if col in numeric_cols:
            empty = vals.isna().sum() + (vals == 0).sum()
        else:
            empty = (vals.fillna("").astype(str) == "").sum()
        assert empty <= len(vals)
        empty_fraction = empty / len(vals)
        empty_fraction = (
            f"{empty_fraction:5.3f}{' ***' if empty_fraction > 0.95 else ''}"
        )
        curr["empty_fraction"] = empty_fraction
        res[col] = curr
    ret = pd.DataFrame(res)
    return ret


def sample_an_sql_table(table_name: str, n_limit: int, cnx: Connection) -> pd.DataFrame:
    """Extract a random sample of a table with an APPROXIMATE size of `n_limit`"""
    n_rows = pd.read_sql(f"SELECT COUNT(*) AS n FROM {table_name} ", cnx).n.iloc[0]
    frac = n_limit / n_rows
    df_sample = pd.read_sql(
        f"SELECT * FROM {table_name} WHERE RAND() < {frac} LIMIT {n_limit}", cnx
    )
    return df_sample


def get_n_rows(cnx: Connection, schema: str, table_name: str):
    """Get the number of rows in a table as a well-formatted string"""
    res = cnx.execute(f"SELECT COUNT(*) from {schema}.{table_name}")
    res = next(res)[0]
    what = f"{schema}.{table_name:30s}"
    ret = f"{what}: {res:12,d} rows"
    return ret


def _actual_create_table(
    cnx: Connection,
    schema: str,
    table_name: str,
    sql: str,
    skip_if_exists: bool = True,
    drop_if_exists: bool = True,
    add_create_table_clause: bool = True,
    verbose=True,
):
    if drop_if_exists:
        if skip_if_exists and verbose:
            print(
                f"Table {schema}.{table_name} exists, `drop_if_exists` is set to True. Dropping it."
            )
        skip_if_exists = False
    if skip_if_exists and table_exists(cnx, table_name, schema):
        if verbose:
            print(f"Table {schema}.{table_name:30s} already exists, skipping")
        return
    if drop_if_exists:
        if verbose:
            print(f"Dropping table {schema}.{table_name:30s}")
        drop_table_if_exists(cnx=cnx, schema=schema, table_name=table_name)
    # Remove remarks
    lines = []
    for l in sql.splitlines():
        l = l.strip()
        if l.startswith("--"):
            continue
        lines.append(l)
    sql = " ".join(lines)
    if add_create_table_clause:
        sql = f"""
            CREATE TABLE {schema}.{table_name} AS (
                {sql}
            )
        """
    print(f"Creating {schema}.{table_name}")
    cnx.execute(sql)
    print(f"Finished creating {schema}.{table_name}")


def table_exists(cnx: Connection, table_name: str, schema_name: str = None) -> bool:
    df_tables = get_tables_and_views(cnx, schema_name)
    return table_name in df_tables.table_name.values


def get_table_summary(cnx: Connection, schema: str, table_name: str) -> str:
    """Get a nicely formatted summary of a table as a string"""
    ret = [get_n_rows(cnx, schema, table_name)]
    df = sample_an_sql_table(
        table_name=f"{schema}.{table_name}", n_limit=10_000, cnx=cnx
    )
    df_sanity = sanity_check_df(df)
    df = pd.concat(
        [
            df.tail(),
            pd.DataFrame([["..."] * df.shape[1]], columns=df.columns, index=["..."]),
            df_sanity,
        ]
    )
    ret.append(df.to_markdown())
    ret = "\n".join(ret)
    return ret
