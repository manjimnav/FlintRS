package es.dmr.uimp

/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.apache.flink.streaming.api.scala._
import java.util.Properties
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer
import org.apache.flink.api.common.serialization.SimpleStringSchema
import org.apache.flink.streaming.util.serialization.JSONKeyValueDeserializationSchema
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.node.ObjectNode
import org.apache.flink.streaming.api.windowing.time.Time
import org.apache.flink.streaming.api.functions.co.ProcessJoinFunction
import org.apache.flink.util.Collector
import org.apache.flink.streaming.api.TimeCharacteristic
import org.pmml4s.model.Model
import play.api.libs.json.Json
//import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.ObjectMapper
import scala.tools.asm.TypeReference
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.ObjectMapper
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.JsonNode
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaProducer
import scala.util.parsing.json.JSONObject

/**
 * Skeleton for a Flink Job.
 *
 * For a full example of a Flink Job, see the WordCountJob.scala file in the
 * same package/directory or have a look at the website.
 *
 * You can also generate a .jar file that you can submit on your Flink
 * cluster. Just type
 * {{{
 *   sbt clean assembly
 * }}}
 * in the projects root directory. You will find the jar in
 * target/scala-2.11/Flink\ Project-assembly-0.1-SNAPSHOT.jar
 *
 */
object Job {
  def main(args: Array[String]): Unit = {
    
    // set up the execution environment
    val env = StreamExecutionEnvironment.getExecutionEnvironment
    env.setStreamTimeCharacteristic(TimeCharacteristic.EventTime)
    
    /**
     * Constants variables
     */
	
println(args(0))
    val token = args(0)

    val model = Model.fromFile("nn_model.pmml")
    
    val objectMapper= new ObjectMapper()
    
    val bootstrapServer = "bigdatamaster2019.dataspartan.com:19093"
    
    // Kafka consumer and producers initialization
    val properties = new Properties()
    properties.setProperty("bootstrap.servers", bootstrapServer)

    val consumerDemographic = new FlinkKafkaConsumer[ObjectNode](
      java.util.regex.Pattern.compile("topic_demographic"),
      new JSONKeyValueDeserializationSchema(false),
      properties)
      
   val consumerHistoric = new FlinkKafkaConsumer[ObjectNode](
      java.util.regex.Pattern.compile("topic_historic"),
      new JSONKeyValueDeserializationSchema(false),
      properties)
      
   val producerPredictions = new FlinkKafkaProducer[String](
        "bigdatamaster2019.dataspartan.com:29093", 
//       "localhost:9092",
        "topic_student_prediction", 
//       "test",
        new SimpleStringSchema)  
   
    /**
     * Functions
     */
    val processFunction = new ProcessJoinFunction[ObjectNode, ObjectNode, Tuple2[String, Map[String, Any]]] {
        override def processElement(left: ObjectNode, right: ObjectNode, ctx: ProcessJoinFunction[ObjectNode, ObjectNode, Tuple2[String, Map[String, Any]]]#Context, out: Collector[Tuple2[String, Map[String, Any]]]): Unit = {
          var userEntry = left.findValue("value").asInstanceOf[ObjectNode] // Get the demographic content
          val historicValue = right.findValue("value") // Get historic content
          
          // Create new object with model inputs
          var modelInputs = objectMapper.createObjectNode(); 
          modelInputs.set("age", userEntry.findValue("age"));
          modelInputs.set("woman", userEntry.findValue("woman"))
          modelInputs.setAll(historicValue.findValue("products").asInstanceOf[ObjectNode])
          
          // Convert object to Map
          val modelArrayEntry = Json.parse(modelInputs.toString()).as[Map[String, Float]]
          println(modelArrayEntry)
          // Return (uuid, input tuple)
          out.collect((userEntry.findValue("uuid").asText(), modelArrayEntry)); 
        }
      }
    
    val makePrediction = (inputPred: Map[String, Any])=> model.predict(inputPred).get("predicted_label").get//.maxBy(_._2.asInstanceOf[Double])._1.charAt(12) // Function that obtains the prediction
    
    /**
     * Pipeline
     */
    val streamDemographic = env.addSource(consumerDemographic)
    
    val streamHistoric = env.addSource(consumerHistoric)

    streamDemographic
    .keyBy(demographicEntry=>demographicEntry.findValue("value").findValue("uuid").asText()) // Obtain the uuid key of demographic information
    .intervalJoin(streamHistoric.keyBy(historicEntry=>historicEntry.findValue("value").findValue("uuid").asText())) // Make join with historic stream obtaining the uuid key
    .between(Time.seconds(-3), Time.seconds(3)).process(processFunction) // Make join within an interval of 3 seconds before and 3 seconds after
    .map(modelInput=>Map("uuid"->modelInput._1.toInt, "value"-> makePrediction(modelInput._2), "token"-> token)) // Construct the response map
//    .map(e=>println(e))
        .map(prediction => JSONObject(prediction).toString()) // Convert map to JSON string
    .addSink(producerPredictions) // Publish to kafka producer
    
    // execute program
    env.execute("Flink Scala API Skeleton")
  }
}
