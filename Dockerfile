FROM maven:3.9.6-eclipse-temurin-17 AS builder
WORKDIR /app

COPY pom.xml .
RUN mvn dependency:go-offline -B

COPY src ./src
RUN mvn clean package -DskipTests

FROM eclipse-temurin:17-jdk
WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/JavaVisualPavement-1.0-SNAPSHOT.jar app.jar

ENV JAVA_OPTS="-Xms2g -Xmx4g -Dorg.nd4j.linalg.cpu.javacpp.maxbytes=4g"

CMD ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]