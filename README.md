# CS230 DDSP Music Source Separation Project
Group Members: Emily Kuo, Samantha Long, Sneha Shah

To install requirements:
<pre><code>pip install -r requirements.txt</code></pre>

To ssh into the AWS instance:
<pre><code>ssh -i "~/.ssh/cs230.pem" ubuntu@ec2-54-149-20-20.us-west-2.compute.amazonaws.com</code></pre>

To run jupyter notbooke on the AWS instance:
<pre><code>ssh -i "~/.ssh/cs230.pem" -fNL 9000:localhost:8888 ubuntu@ec2-54-149-20-20.us-west-2.compute.amazonaws.com</code></pre>

Install ffmpeg through this command (not pip install):
<pre><code>conda install -c conda-forge ffmpeg</code></pre>
