import pymysql
import random

sql = {
    'host': 'localhost',
    'user': 'military',
    'password': '1234',
    'db': 'military'
}
#
# def generate_random_coordinates():
#     # 무작위 위도 및 경도 생성
#     latitude = round(random.uniform(-90, 90), 6)
#     longitude = round(random.uniform(-180, 180), 6)
#     return latitude, longitude

def is_coordinate_duplicate(cursor, latitude, longitude):
    # 좌표값이 이미 데이터베이스에 존재하는지 확인
    select_query = "SELECT COUNT(*) FROM coordinates WHERE latitude = %s AND longitude = %s"
    cursor.execute(select_query, (latitude, longitude))
    result = cursor.fetchone()[0]
    return result > 0

def conn(): # DB 연결
    return pymysql.connect(host=sql['host'], user=sql['user'], password=sql['password'], db=sql['db'])

def save_enemies(lat,lng):
    DB = conn()
    cursor = DB.cursor()

    insert_query = f"INSERT INTO coordinates (latitude, longitude) VALUES ({lat}, {lng})"

    cursor.execute(insert_query)
    DB.commit()
    cursor.close()

def save_current(lat,lng):
    DB = conn()
    cursor = DB.cursor()

    insert_query = f"INSERT INTO location (latitude, longitude) VALUES ({lat}, {lng})"
    cursor.execute(insert_query)

    cursor.execute("SELECT COUNT(*) FROM location")
    total_count = cursor.fetchone()[0]

    if total_count > 50:
        delete_query = "DELETE FROM location ORDER BY id LIMIT %s"
        excess_count = total_count - 50
        cursor.execute(delete_query, excess_count)

    DB.commit()
    cursor.close()
if __name__ == '__main__':
    # print(generate_random_coordinates())
    for i in range(100):
        save_current(3.32 + i, i / 10 + 0.01)
    # print(save_enemies(39.2334,128.4256))