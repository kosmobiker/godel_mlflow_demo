# Configure the AWS provider
provider "aws" {
  region = "eu-west-1"
}

# Create a security group for the EC2 instance
resource "aws_security_group" "ec2_sg" {
  name_prefix = "ec2_sg"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Create the EC2 instance
resource "aws_instance" "mlflow-tracking-server" {
  ami = "ami-09dd5f12915cfb387"
  instance_type = "t2.micro"
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]
  key_name = "godel_keypair"
  tags = {
    Name = "mlflow"
  }
}

# Create a security group for the RDS instance
resource "aws_security_group" "rds_sg" {
  name_prefix = "rds_sg"
  
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Create the RDS instance
resource "aws_db_instance" "mlflow-backend-db" {
  allocated_storage = 20
  engine = "postgres"
  engine_version = "14.6"
  instance_class = "db.t4g.micro"
  db_name = "mlflow"
  username = "mlflow"
  password = "password123"
  skip_final_snapshot = true
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
}
