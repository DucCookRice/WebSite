###Create T7 feature 
import pyspark.sql.functions as F
from pyspark.sql.window import Window


lumpy_data = spark.read.parquet( "abfss://project-frt@useelakestorage.dfs.core.windows.net/datasource/clu/FRT.TransHis_lumpy_clu/")
smooth_data = spark.read.parquet( "abfss://project-frt@useelakestorage.dfs.core.windows.net/datasource/clu/FRT.TransHis_smooth_clu/")
erratic_data = spark.read.parquet( "abfss://project-frt@useelakestorage.dfs.core.windows.net/datasource/clu/FRT.TransHis_erratic_clu/")
intermittent_data = spark.read.parquet( "abfss://project-frt@useelakestorage.dfs.core.windows.net/datasource/clu/FRT.TransHis_intermittent_clu/")
all_cluster_data = lumpy_data.union(smooth_data).union(erratic_data).union(intermittent_data)


lumpy_predict = spark.read.parquet("abfss://project-frt@useelakestorage.dfs.core.windows.net/datasource/raw/FRT.TransHis/sample_data/predict_lumpy_result_jar_spark_par/")
smooth_predict = spark.read.parquet("abfss://project-frt@useelakestorage.dfs.core.windows.net/datasource/raw/FRT.TransHis/sample_data/predict_smooth_result_jar_spark_par/")
erratic_predict = spark.read.parquet("abfss://project-frt@useelakestorage.dfs.core.windows.net/datasource/raw/FRT.TransHis/sample_data/predict_erratic_result_jar_spark_par/")
intermittent_predict = spark.read.parquet("abfss://project-frt@useelakestorage.dfs.core.windows.net/datasource/raw/FRT.TransHis/sample_data/predict_intermittent_result_jar_spark_par/")

predict_data = lumpy_predict.union(smooth_predict).union(erratic_predict).union(intermittent_predict).withColumn("DATE",F.col("SNAPSHOT_DATE") - 1)



window_7T = Window.partitionBy("ITEMCODE").orderBy("TRANS_DATE").rowsBetween(-3,Window.currentRow-1)
data_7T = all_cluster_data.withColumn("ITEMCODE",F.substring(F. col("ID"),10,20))\
                    .withColumn("WHSCODE",F.substring(F.col("ID"),1,8))\
                    .groupBy(["TRANS_DATE","ITEMCODE"])\
                    .agg(F.sum("QTY").alias("SYSTEM_TRANS_HIS"))\
                    .sort(["ITEMCODE","TRANS_DATE"])\
                    .withColumn("7_T",F.sum(F.col("SYSTEM_TRANS_HIS"))\
                    .over(window_7T))\
                    .fillna(0)
# data_7T = data_7T.groupBy(["TRANS_DATE","ITEMCODE"]).agg(F.sum("QTY").alias("SYSTEM_TRANS_HIS")).sort(["ITEMCODE","TRANS_DATE"]).withColumn("7_T",F.sum(F.col("SYSTEM_TRANS_HIS")).over(window_7T)).fillna(0)

window = Window.partitionBy("ID").orderBy("TRANS_DATE")

inv_path = "abfss://project-frt@useelakestorage.dfs.core.windows.net/datasource/stg/FRT.Inventory_full/"
inv_data = spark.read.parquet(inv_path,header = True)

inv_data = inv_data.withColumn("DATE",F.substring(F.col("TRANS_DATE"),1,10)).drop("W_INSERT_DT").drop("SNAPSHOT_DATE")\
        .withColumn("ID",F.concat(F.col("WAREHOUSE_CODE"),F.lit("_"),F.col("PRODUCT_CODE"))).withColumn('Number',row_number().over(window))

data_kho_tong = inv_data.filter(F.col("WAREHOUSE_CODE") =="80001010" )
data_kho_tong = data_kho_tong.drop("ID").drop("SHOP_CODE").drop("Number").drop("WAREHOUSE_CODE").drop("UNIT_PRICE").drop("TOTAL_PRICE").drop("QTY_L1").drop("TRANS_DATE")

optimized_input = predict_data.join(data_kho_tong,["PRODUCT_CODE","DATE"],"leftouter")
optimized_input.show()

# optimized_input_pandas = optimized_input.toPandas()