-- MySQL dump 10.13  Distrib 8.0.40, for macos14 (x86_64)
--
-- Host: 127.0.0.1    Database: tractive
-- ------------------------------------------------------
-- Server version	8.4.3

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `movies`
--

DROP TABLE IF EXISTS `movies`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `movies` (
  `id` int NOT NULL AUTO_INCREMENT,
  `imdb_key` varchar(9) DEFAULT NULL,
  `title` varchar(150) DEFAULT NULL,
  `release_date` int NOT NULL,
  `genre` varchar(150) DEFAULT NULL,
  `director` varchar(150) DEFAULT NULL,
  `director_gender` varchar(150) DEFAULT NULL,
  `trailer_obj` varchar(150) DEFAULT NULL,
  `framerate` float DEFAULT '24',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=51 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `movies`
--

LOCK TABLES `movies` WRITE;
/*!40000 ALTER TABLE `movies` DISABLE KEYS */;
INSERT INTO `movies` VALUES (1,'tt2267998','Gone Girl',2014,'Drame, Thriller, MystÃ¨re','David Fincher','Men','likely',23.9759),(2,'tt0790636','Dallas Buyers Club',2013,'Biographique, Drame','Jean-Marc VallÃ©e','Men',NULL,23.976),(3,'tt1798709','Her',2013,'Drame, Romantique, Science-fiction','Spike Jonze','Men',NULL,23.976),(4,'tt1045658','Silver Linings Playbook',2012,'Comedie, Drame, Romantique','David O. Russell','Men','sure',23.976),(5,'tt1907668','Flight',2012,'Drame, Thriller','Robert Zemeckis','Men',NULL,23.976),(6,'tt1189340','The Lincoln Lawyer',2011,'Policier, Drame, MystÃ¨re','Brad Furman','Men','sure',24),(7,'tt1568346','The Girl with the Dragon Tattoo',2011,'Policier, Drame, MystÃ¨re','David Fincher','Men','sure',23.976),(8,'tt1570728','Crazy,Stupid,Love',2011,'Comedie, Drame, Romantique','Glenn Ficarra & John Requa','Men & Men','sure',23.976),(9,'tt1499658','Horrible Bosses',2011,'Comedie, Policier','Seth Gordon','Men',NULL,23.976),(10,'tt1632708','Friends with Benefits',2011,'Comedie, Romantique','Will Gluck','Men','sure',23.976),(11,'tt1454029','The Help',2011,'Drame, nc','Tate Taylor','Men',NULL,23.976),(12,'tt1285016','The Social Network',2010,'Biographique, Drame','David Fincher','Men','sure',23.976),(13,'tt1385826','The Adjustment Bureau',2010,'Romantique, Science-fiction, Science-fiction','George Nolfi','Men','likely',23.976),(14,'tt1193138','Up in the Air',2009,'Comedie, Drame, Romantique','Jason Reitman','Men',NULL,23.976),(15,'tt1142988','The Ugly Truth',2009,'Comedie, Romantique','Robert Luketic','Men','sure',23.976),(16,'tt0822832','Marley & Me',2008,'Drame, Famille','David Frankel','Men',NULL,23.976),(17,'tt1013753','Milk',2008,'Biographique, Drame, Historique','Gus Van Sant','Men','likely men ?',23.976),(18,'tt0455824','Australia',2008,'Aventure, Drame, Romantique','Baz Luhrmann','Men','likely',23.976),(19,'tt1010048','Slumdog Millionaire',2008,'Policier, Drame, Romantique','Dan Boyle & Leveleen Tandan','Men & Women','sure',25),(20,'tt0970416','The Day the Earth Stood Still',2008,'Aventure, Drame, Science-fiction','Scott Derrickson','Men','likely',23.976),(21,'tt0988595','27 Dresses',2008,'Comedie, Romantique','Anne Fletcher','Women','sure',25),(22,'tt0467406','Juno',2007,'Comedie, Drame','Jason Reitman','Men','likely men ?',23.976),(23,'tt0478311','Knocked Up',2007,'Comedie, Romantique','Judd Apatow','Men','sure likely',25),(24,'tt0388795','Brokeback Mountain',2005,'Drame, Romantique','Ang Lee','Men','likely men ?',23.976),(25,'tt0416320','Match Point',2005,'Drame, Romantique, Thriller','Woody Allen','Men','sure',23.976),(26,'tt0375679','Crash',2004,'Policier, Drame, Thriller','Paul Haggis','Men',NULL,24),(27,'tt0317198','Bridget Jones: The Edge of Reason',2004,'Comedie, Drame, Romantique','Beeban Kidron','Men','likely',23.976),(28,'tt0307987','Bad Santa',2003,'Comedie, Policier, Drame','Terry Zwigoff','Men','sure likely',23.976),(29,'tt0286106','Signs',2002,'Drame, Mystere, Science-fiction','M. Night Shyamalan','Men',NULL,23.976),(30,'tt0240772','Ocean\'s Eleven',2001,'Policier, Thriller','Steven Soderbergh','Men','likely',23.9445),(31,'tt0241527','Harry Potter and the Sorcerer\'s Stone',2001,'Aventure, Famille, Fantastique','Chris Columbus','Men',NULL,25),(32,'tt0212338','Meet the Parents',2000,'Comedie, Romantique','Jay Roach','Men',NULL,25),(33,'tt0146882','High Fidelity',2000,'Comedie, Drame, Musical','Stephen Frears','Men',NULL,23.976),(34,'tt0167404','The Sixth Sense',1999,'Drame, Mystere, Thriller','M. Night Shyamalan','Men',NULL,25),(35,'tt0147800','10 Things I Hate About You',1999,'Comedie, Drame, Romantique','Gil Junger','Men','sure',23.976),(36,'tt0118715','The Big Lebowski',1998,'Comedie, Policier','Ethan Coen & Joel Coen','Men & Men','sure',23.976),(37,'tt0119822','As Good as It Gets',1997,'Comedie, Drame, Romantique','James L. Brooks','Men','sure likely',23.976),(38,'tt0118842','Chasing Amy',1997,'Comedie, Drame, Romantique','Kevin Smith','Men','likely',23.976),(39,'tt0120338','Titanic',1997,'Drame, Romantique','James Cameron','Men','likely',23.976),(40,'tt0116695','Jerry Maguire',1996,'Comedie, Drame, Romantique','Cameron Crowe','Men',NULL,23.976),(41,'tt0114924','While You Were Sleeping',1995,'Comedie, Drame, Romantique','Jon Turteltaub','Men',NULL,23.976),(42,'tt0109830','Forrest Gump',1994,'Drame, Romantique','Robert Zemeckis','Men',NULL,23.976),(43,'tt0109831','Four Weddings and a Funeral',1994,'Comedie, Drame, Comedie','Mike Newell','Men','sure',25),(44,'tt0110912','Pulp Fiction',1994,'Policier, Drame','Quentin Tarantino','Men','sure',23.976),(45,'tt0108160','Sleepless in Seattle',1993,'Comedie, Drame, Romantique','Nora Ephron','Women','sure ling',23.976),(46,'tt0106918','The Firm',1993,'Drame, Mystere, Thriller','Sydney Pollack','Men',NULL,23.976),(47,'tt0100405','Pretty Woman',1990,'Comedie, Romantique','Garry Marshall','Men','sure',23.976),(48,'tt0097576','Indiana Jones and the Last Crusade',1989,'Action, Aventure','Steven Spielberg','Men','sure',23.976),(49,'tt0073486','One Flew Over the Cuckoo\'s Nest',1975,'Drame, nc','Milos Forman','Men','sure',23.976),(50,'tt0068646','The Godfather',1972,'Policier, Drame','Francis Ford Coppola','Men',NULL,23.976);
/*!40000 ALTER TABLE `movies` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2026-02-14 17:55:20
