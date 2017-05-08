package bike

//time lib
import org.joda.time._
//spark lib
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._

//log
import org.apache.log4j._

//MLlib lib
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.linalg.Vectors
//random forest
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import java.text.DecimalFormat;

object BikeShareRegressionRF {
  case class BikeShareEntity(instant: String, dteday: String, season: Double, yr: Double, mnth: Double,
                             hr: Double, holiday: Double, weekday: Double, workingday: Double, weathersit: Double, temp: Double,
                             atemp: Double, hum: Double, windspeed: Double, casual: Double, registered: Double, cnt: Double)
  def main(args: Array[String]): Unit = {
    setLogger
    val doTrain = (args != null && args.length > 0 && "Y".equals(args(0)))
    val sc = new SparkContext(new SparkConf().setAppName("BikeRegressionRF").setMaster("local[*]"))

    println("============== preparing data ==================")
    val (trainData, validateData) = prepare(sc)
    trainData.persist(); validateData.persist();

    val cateInfo = getCategoryInfo()
    if (!doTrain) {
      println("============== train Model (CateInfo)==================")
      val (modelC, durationC) = trainModel(trainData, "variance", 10, 30, 50, cateInfo)
      val rmseC = evaluateModel(validateData, modelC)
      println("validate rmse(CateInfo)=%f".format(rmseC))
      println("---predict Data---")
      PredictData(sc, modelC)
    } else {
      println("============== tuning parameters(CateInfo) ==================")
      tuneParameter(trainData, validateData, cateInfo)

    }

    trainData.unpersist(); validateData.unpersist();

  }

  def prepare(sc: SparkContext): (RDD[LabeledPoint], RDD[LabeledPoint]) = {
    val rawDataWithHead = sc.textFile("data/hour.csv")
    val rawDataNoHead = rawDataWithHead.mapPartitionsWithIndex { (idx, iter) => { if (idx == 0) iter.drop(1) else iter } }
    val rawData = rawDataNoHead.map { x => x.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)", -1).map { x => x.trim() } }

    println("read BikeShare Dateset count=" + rawData.count())
    val bikeData = rawData.map { x =>
      BikeShareEntity(x(0), x(1), x(2).toDouble, x(3).toDouble, x(4).toDouble,
        x(5).toDouble, x(6).toDouble, x(7).toDouble, x(8).toDouble, x(9).toDouble, x(10).toDouble,
        x(11).toDouble, x(12).toDouble, x(13).toDouble, x(14).toDouble, x(15).toDouble, x(16).toDouble)
    }

    val lpData = bikeData.map { x =>
      {
        val label = x.cnt
        val features = Vectors.dense(getFeatures(x))
        new LabeledPoint(label, features) //LabeledPoint由label及Vector組成
      }
    }
    //以6:4的比例隨機分割，將資料切分為訓練及驗證用資料
    val Array(trainData, validateData) = lpData.randomSplit(Array(0.8, 0.2))
    (trainData, validateData)
  }

  def PredictData(sc: SparkContext, model: RandomForestModel): Unit = {

    val rawDataWithHeader = sc.textFile("data/hourtest.csv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
    val splitlines = rawData.map(_.split(","))
    println("count：" + splitlines.count.toString())
    println()
    val trainData = splitlines.take(10).map { col =>
      val instant = col(0)
      val season = col(2).toInt
      val yr = col(3).toInt
      val mnth = col(4).toInt
      val hr = col(5).toInt
      val holiday = col(6).toInt
      val weekday = col(7).toInt
      val workingday = col(8).toInt
      val weathersit = col(9).toInt
      val temp = col(10).toDouble
      val atemp = col(11).toDouble
      val hum = col(12)
      val windspeed = col(13)
      val casual = col(14)
      val registered = col(15)

      val df2 = new DecimalFormat(".##") // double decimal 2 ## 就是取兩個數字
      val label = 0
      //進行預測
      val features = col.slice(2, 15).map(_.toDouble)
      val Features = Vectors.dense(features)
      val predict = model.predict(Features).toInt
      val pp = predict - (col(16).toInt)
      var weathersitD = { weathersit match { case 1 => "Clear"; case 2 => "Mist"; case 3 => "Light Snow"; case 4 => "Heavy Rain" } }
      var workingdayD = { workingday match { case 0 => "工作日"; case 1 => "非工作日"; } }
      var weekdayD = { weekday match { case 0 => "星期日"; case 1 => "星期一"; case 2 => "星期二"; case 3 => "星期三"; case 4 => "星期四"; case 5 => "星期五"; case 6 => "星期六"; } }
      var holidayD = { holiday match { case 0 => "非假日"; case 1 => "假日"; } }
      var mnthD = { mnth match { case 1 => "1月"; case 2 => "2月"; ; case 3 => "3月"; case 4 => "4月"; case 5 => "5月"; case 6 => "6月"; case 7 => "7月"; case 8 => "8月"; case 9 => "9月"; case 10 => "10月"; case 11 => "11月"; case 12 => "12月"; } }
      var seasonD = { season match { case 1 => "spring"; case 2 => "summer"; case 3 => "autume"; case 4 => "winter" } }
      var hrD = { hr match { case 0 => "00時"; case 1 => "01時"; case 2 => "02時"; ; case 3 => "03時"; case 4 => "04時"; case 5 => "05時"; case 6 => "06時"; case 7 => "07時"; case 8 => "08時"; case 9 => "09時"; case 10 => "10時"; case 11 => "11時"; case 12 => "12時"; case 13 => "13時"; case 14 => "14時"; case 15 => "15時"; case 16 => "16時"; case 17 => "17時"; case 18 => "18時"; case 19 => "19時"; case 20 => "20時"; case 21 => "21時"; case 22 => "22時"; case 23 => "23時" } }
      println("特徵:" + seasonD + ',' + mnthD + ',' + hrD + ',' + holidayD + ',' + weekdayD + ',' + workingdayD + ',' + weathersitD + ',' + df2.format(temp * 41) + "度" + ' ' + "體感" + (atemp * 50) + "度" + ',' + "濕度" + hum + ',' + "風速" + windspeed + "---------------------" + "Predict:" + predict + "    " + "實際:" + col(16) + "    " + "誤差:" + pp)
    }

  }

  def getFeatures(bikeData: BikeShareEntity): Array[Double] = {
    val featureArr = Array(bikeData.yr, bikeData.season - 1, bikeData.mnth - 1, bikeData.hr,
      bikeData.holiday, bikeData.weekday, bikeData.workingday, bikeData.weathersit - 1, bikeData.temp, bikeData.atemp,
      bikeData.hum, bikeData.windspeed)
    featureArr
  }

  def getCategoryInfo(): Map[Int, Int] = {
    //("yr", 2), ("season", 4), ("mnth", 12), ("hr", 24),
    //("holiday", 2), ("weekday", 7), ("workingday", 2), ("weathersit", 4)
    val categoryInfoMap = Map[Int, Int](( /*"yr"*/ 0, 2), ( /*season*/ 1, 4), ( /*"mnth"*/ 2, 12), ( /*"hr"*/ 3, 24),
      ( /*"holiday"*/ 4, 2), ( /*"weekday"*/ 5, 7), ( /*"workingday"*/ 6, 2), ( /*"weathersit"*/ 7, 4))
    //val categoryInfoMap = Map[Int, Int]()
    categoryInfoMap
  }

  def trainModel(trainData: RDD[LabeledPoint],
                 impurity: String, numTrees: Int, maxDepth: Int, maxBins: Int, catInfo: Map[Int, Int]): (RandomForestModel, Double) = {
    val startTime = new DateTime()

    val model = RandomForest.trainRegressor(trainData, catInfo, numTrees, "auto", impurity, maxDepth, maxBins)
    val endTime = new DateTime()
    val duration = new Duration(startTime, endTime)
    //MyLogger.debug(model.toDebugString)
    (model, duration.getMillis)
  }

  def evaluateModel(validateData: RDD[LabeledPoint], model: RandomForestModel): Double = {
    val scoreAndLabels = validateData.map { data =>
      var predict = model.predict(data.features)

      (predict, data.label)
    }
    val metrics = new RegressionMetrics(scoreAndLabels)
    val rmse = metrics.rootMeanSquaredError
    rmse
  }

  def tuneParameter(trainData: RDD[LabeledPoint], validateData: RDD[LabeledPoint], cateInfo: Map[Int, Int]) = {
    val impurityArr = Array("variance")
    val numTreesArr = Array(3, 5, 10)
    val depthArr = Array(3, 5, 10, 15, 20, 25)
    val binsArr = Array(3, 5, 10, 50, 100, 200)
    val evalArr =
      for (impurity <- impurityArr; numTree <- numTreesArr; maxDepth <- depthArr; maxBins <- binsArr) yield {
        val (model, duration) = trainModel(trainData, impurity, numTree, maxDepth, maxBins, cateInfo)
        val rmse = evaluateModel(validateData, model)
        println("parameter: impurity=%s, numTree=%d, maxDepth=%d, maxBins=%d, rmse=%f"
          .format(impurity, numTree, maxDepth, maxBins, rmse))
        (impurity, numTree, maxDepth, maxBins, rmse)
      }
    val bestEvalAsc = (evalArr.sortBy(_._5))
    val bestEval = bestEvalAsc(0)
    println("best parameter: impurity=%s, numTree=%d, maxDepth=%d, maxBins=%d, rmse=%f"
      .format(bestEval._1, bestEval._2, bestEval._3, bestEval._4, bestEval._5))
  }

  def setLogger = {
    Logger.getLogger("org").setLevel(Level.OFF) //mark for MLlib INFO msg
    Logger.getLogger("com").setLevel(Level.OFF)
    Logger.getLogger("io").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.ALL);
  }
}